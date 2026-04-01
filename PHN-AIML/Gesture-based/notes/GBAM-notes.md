# GBAM — Gesture-Based Audio Modulator
## Study Notes

---

## 1. What This Project Does

You raise your **index finger** → pitch goes up.  
You move your **thumb** left/right → playback speed changes.  
You raise your **middle finger** → volume goes up.

The pipeline is:

```
Webcam → MediaPipe (hand landmarks) → Gesture values → DSP → Speaker
```

The project has two modes:
- **FILE mode** — plays a song file (Bheegi.mp3) and modulates it
- **MIC mode** — takes live mic input and modulates your voice in real-time

---

## 2. Why The Old Code Was Stuttering (Root Cause Analysis)

### The Core Conflict

There are two threads racing each other:

| Thread | Job | Speed |
|---|---|---|
| `playback_loop` | Process audio with librosa | ~150ms per chunk |
| `play_audio` | Write chunk to speaker | needs new chunk every ~37ms |

The play thread was asking for a new chunk every 37ms. The processing thread took 150ms to produce one. So the play thread **starved** — ran out of audio — and you heard silence/stuttering.

### Why CHUNK_SIZE Matters

```
Chunk Duration = CHUNK_SIZE / SAMPLE_RATE

CHUNK_SIZE = 8192  →  8192 / 22050 = ~37ms
CHUNK_SIZE = 163840 →  163840 / 22050 = ~743ms
```

| Chunk Size | Duration | librosa time | Result |
|---|---|---|---|
| 4096 | 18ms | 150ms | Heavy stutter |
| 8192 | 37ms | 150ms | Stutter |
| 16384 | 74ms | 150ms | Borderline |
| 163840 | 743ms | 150ms | Smooth BUT 743ms gesture lag |

Bigger chunk = more audio pre-buffered = fewer starves. But bigger chunk also means gesture changes take longer to be heard. A 743ms lag is unusable as a live controller.

### It's Not a Linux Problem

Windows audio drivers have more aggressive internal buffering, which *hides* the lag rather than fixing it. Your friends' code had the same architectural problem — Windows just concealed it.

---

## 3. All the Problems in the Old Code

### Problem 1 — Wrong DSP Library
`librosa.effects.pitch_shift` uses a **phase vocoder** internally. Phase vocoder is a batch signal processing algorithm designed for analysis, not real-time use. It processes an entire buffer at once and is inherently slow (~80–200ms per chunk).

### Problem 2 — Race Condition on `gesture` Dict
```python
# Thread A (camera) writes:
gesture["pitch"] = 12.0

# Thread B (audio) reads:
pitch_steps = gesture["pitch"]
```
No lock. On Linux, the GIL (Global Interpreter Lock) doesn't protect this reliably across threads. On Windows the GIL behavior hides it. The fix is a `threading.Lock()` around every read/write.

### Problem 3 — No Gesture Smoothing
When you move your hand, the raw landmark coordinate can jump from `cy=50` to `cy=400` in a single frame. That maps directly to a pitch jump from `+12` to `-12` semitones. Even fast DSP will click on a step that large.

### Problem 4 — Running Two Heavy Operations Together
Some versions (GBAM-4, GBAM-2) ran both `pitch_shift` **and** `time_stretch` in the same chunk. Two full phase-vocoder passes = ~300ms per chunk.

---

## 4. The Fixes in GBAM-v5

### Fix 1 — Replace librosa with pedalboard
```python
from pedalboard import Pedalboard, PitchShift

board = Pedalboard([PitchShift(semitones=0)])
output = board(audio_chunk, sample_rate)  # ~5ms, not 150ms
```

`pedalboard` is Spotify's audio processing library. It wraps professional C++ DSP algorithms. `PitchShift` inside it uses a real-time-capable algorithm, not a batch phase vocoder.

If `pedalboard` is not installed, `GBAM-v5.py` falls back to the **resampling trick**:

```
pitch_ratio = 2 ^ (semitones / 12)

# Example: +12 semitones = ratio 2.0 (one octave up)
# Resample the audio as if it was recorded at 2x the sample rate,
# then play it back at normal rate → sounds pitched up

resampled = librosa.resample(chunk, orig_sr=int(sr * ratio), target_sr=sr)
```

This is ~10× cheaper than the phase vocoder. Quality is lower at large shifts, but acceptable for a demo.

### Fix 2 — Thread-Safe GestureState with EMA Smoothing

```python
class GestureState:
    def update(self, **kwargs):
        with self._lock:
            self._smoothed[key] = alpha * raw_value + (1 - alpha) * self._smoothed[key]
```

Two things happening here:
1. **Lock** — guarantees the camera thread and audio thread never touch the same data simultaneously
2. **EMA (Exponential Moving Average)** — smooths out sudden jumps in gesture values

EMA formula:  `smooth[t] = α × raw[t] + (1−α) × smooth[t−1]`

- `α = 0.12` (our setting) — responds in ~8 frames, never clicks
- `α = 1.0` — no smoothing, raw values (old behaviour, clicks)
- `α = 0.01` — very smooth, feels unresponsive

### Fix 3 — Speed via Resampling (not time_stretch)

Old: `librosa.effects.time_stretch(chunk, rate)` → phase vocoder, slow  
New: `librosa.resample(chunk, orig_sr=int(sr*rate), target_sr=sr)` → simple rate change, fast

The trade-off: `time_stretch` preserves pitch while changing speed (proper time stretching). The resample trick changes both speed and pitch together — but since pitch shift is applied separately anyway, this doesn't matter here.

### Fix 4 — Crossfade Between Chunks

Even with fast DSP, consecutive chunks can end at different amplitude levels. When two chunks are joined at a discontinuity, you hear a click. Crossfade blends the tail of chunk N into the head of chunk N+1:

```
result[:fade_len] = prev_chunk[-fade_len:] * fade_out  +  curr_chunk[:fade_len] * fade_in

fade_out = [1.0, 0.97, 0.94, ... 0.0]   (linear ramp down)
fade_in  = [0.0, 0.03, 0.06, ... 1.0]   (linear ramp up)
```

---

## 5. OOP Architecture

```
GestureAudioModulator       (orchestrator — owns everything, runs event loop)
├── GestureState            (shared data store — thread-safe + EMA)
├── AudioEngine             (audio I/O + DSP pipeline)
│   └── DSP                 (stateless processing helpers)
└── GestureController       (webcam + MediaPipe + landmark mapping)
```

### Why This Structure?

**Single Responsibility** — each class has one job:
- `GestureState` only stores and smooths parameters
- `DSP` only transforms audio
- `AudioEngine` only manages audio threads
- `GestureController` only handles vision

**Testability** — you can unit test `DSP.pitch_shift()` without a webcam.

**Extendability** — adding reverb? Add it to `DSP`. Adding a new finger gesture? Only `GestureController` changes.

---

## 6. Threading Model

```
Main Thread          GestureController.read_frame()
                     │
                     └→  updates GestureState (with lock)

ProcessLoop Thread   AudioEngine._process_loop()      [FILE mode only]
                     │
                     └→  reads GestureState (with lock)
                         applies DSP
                         writes to _output_buf (with lock)

PlayLoop Thread      AudioEngine._play_loop()
                     │
                     └→  reads _output_buf (with lock)
                         writes to sounddevice OutputStream

sounddevice Thread   AudioEngine._mic_callback()       [MIC mode only]
                     │
                     └→  reads from mic
                         applies DSP
                         writes to _output_buf (with lock)
```

The **double buffer** (`_output_buf`) is the key to smooth playback:
- Producer (process/mic) writes to it whenever a chunk is ready
- Consumer (play) reads from it every ~185ms
- If producer is slow, consumer re-plays the last chunk (brief repeat)
- A repeat is much less noticeable than silence

---

## 7. MediaPipe Landmark Reference

MediaPipe returns 21 landmark points per hand. We use:

```
4  = Thumb tip        → controls Speed  (X position)
8  = Index finger tip → controls Pitch  (Y position)
12 = Middle finger tip → controls Volume (Y position)
```

Landmark coordinates are **normalized** (0.0 to 1.0), where:
- `x = 0.0` = left edge of frame
- `x = 1.0` = right edge of frame
- `y = 0.0` = top of frame
- `y = 1.0` = bottom

We map these with `np.interp`:
```python
pitch = np.interp(index_y, [0.1, 0.9], [+12.0, -12.0])
# index at top of frame (y≈0.1) → +12 semitones
# index at bottom       (y≈0.9) → -12 semitones
```

The `[0.1, 0.9]` (not `[0, 1]`) gives a small dead zone at the edges so extreme values aren't triggered accidentally.

---

## 8. How to Run

```bash
# Activate your venv first
source venv/bin/activate

# Install pedalboard (recommended, one-time)
pip install pedalboard

# Song file mode (default)
python GBAM-v5.py file Bheegi.mp3

# Live mic mode
python GBAM-v5.py mic

# Check your audio devices if you need to change INPUT/OUTPUT_DEVICE:
python -c "import sounddevice; print(sounddevice.query_devices())"
```

---

## 9. What Can Be Improved Next

- **Hot-swap mode** — press `M` to toggle between FILE and MIC without restarting
- **More fingers** — ring finger for reverb, pinky for echo
- **Two hands** — left hand = effects, right hand = playback control
- **Visual feedback** — vertical bars showing pitch/speed/volume levels
- **Reverb / Delay** — `pedalboard` supports these natively (`Reverb`, `Delay` classes)
- **Pitch quantization** — snap to musical notes instead of continuous semitones

---

*Last updated: GBAM-v5 — OOP refactor with pedalboard DSP*
