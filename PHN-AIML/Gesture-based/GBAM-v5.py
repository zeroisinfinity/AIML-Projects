"""
GBAM-v5.py  —  Gesture-Based Audio Modulator  (Clean OOP Version)
==================================================================
Modes:
    python GBAM-v5.py file Bheegi.mp3   →  play a song file
    python GBAM-v5.py mic               →  process live mic input

Controls:
    Index finger  UP / DOWN     →  Pitch   (±12 semitones)
    Thumb         LEFT / RIGHT  →  Speed   (0.5x – 2.0x)
    Middle finger UP / DOWN     →  Volume  (0.0 – 1.5)
    Q                           →  Quit
"""

# ── silence noisy C++ loggers before any import ──────────────────────────────
import os
os.environ['GRPC_VERBOSITY']    = 'ERROR'
os.environ['GLOG_minloglevel']  = '2'
os.environ['QT_LOGGING_RULES']  = '*.debug=false'
os.environ['QT_QPA_PLATFORM']   = 'xcb'

import sys
import enum
import threading
import time
import warnings
warnings.filterwarnings("ignore")

import cv2
import mediapipe as mp
import numpy as np
import sounddevice as sd
import librosa

# ── optional fast DSP backend ─────────────────────────────────────────────────
try:
    from pedalboard import Pedalboard, PitchShift
    HAS_PEDALBOARD = True
    print("✅ pedalboard found — using fast C++ pitch shift")
except ImportError:
    HAS_PEDALBOARD = False
    print("⚠️  pedalboard not installed — using resampling fallback")
    print("   (install with: pip install pedalboard)")


# =============================================================================
# ENUMS & CONFIG
# =============================================================================

class AudioMode(enum.Enum):
    FILE = "file"
    MIC  = "mic"


class Config:
    """Single source of truth for all tunable constants."""

    # Audio
    SAMPLE_RATE   : int   = 22050
    CHUNK_SIZE    : int   = 4096      # ~185 ms per chunk — low latency
    CROSSFADE_LEN : int   = 512       # samples blended between chunks
    OUTPUT_DEVICE : int   = 13
    INPUT_DEVICE  : int   = 13

    # Gesture smoothing — Exponential Moving Average factor
    #   0.05 = very smooth / sluggish
    #   0.30 = responsive but may click on fast moves
    SMOOTHING : float = 0.12

    # Parameter ranges
    PITCH_RANGE  : tuple = (-12.0, 12.0)   # semitones
    SPEED_RANGE  : tuple = (0.5,   2.0)    # multiplier
    VOLUME_RANGE : tuple = (0.0,   1.5)    # linear gain


# =============================================================================
# GESTURE STATE  —  thread-safe + EMA smoothing
# =============================================================================

class GestureState:
    """
    Shared, thread-safe store for gesture parameters.

    The EMA (Exponential Moving Average) is the key fix for audio clicks:
        smooth[t] = α * raw[t] + (1−α) * smooth[t−1]
    Without this, pitch can jump from +12 → -12 in one frame and cause
    an audible pop even if the DSP is fast.
    """

    _DEFAULTS = {"pitch": 0.0, "speed": 1.0, "volume": 1.0}

    def __init__(self, smoothing: float = Config.SMOOTHING):
        self._lock     = threading.Lock()
        self._alpha    = smoothing
        self._raw      = dict(self._DEFAULTS)
        self._smoothed = dict(self._DEFAULTS)

    def update(self, **kwargs) -> None:
        """Called from the main (camera) thread."""
        with self._lock:
            for key, val in kwargs.items():
                if key not in self._raw:
                    continue
                v = float(val)
                self._raw[key]      = v
                self._smoothed[key] = (
                    self._alpha * v +
                    (1.0 - self._alpha) * self._smoothed[key]
                )

    def get(self) -> dict:
        """Called from the audio processing thread — returns smoothed copy."""
        with self._lock:
            return dict(self._smoothed)


# =============================================================================
# DSP  —  all audio processing in one place
# =============================================================================

class DSP:
    """
    Stateless audio processing helpers.

    Pitch shift strategy
    ────────────────────
    pedalboard available  →  Pedalboard(PitchShift)
        C++ phase vocoder, ~5 ms per chunk, designed for real-time use.

    pedalboard NOT available  →  Resampling trick
        Play back at a different sample rate then resample to target.
        Not artifact-free at large shifts but 10× cheaper than librosa's
        phase vocoder.  Formula:  ratio = 2^(semitones/12)

    Speed change strategy
    ─────────────────────
    Resample with librosa (no phase vocoder, just rate change).
    Much faster than time_stretch because it doesn't preserve pitch.
    """

    def __init__(self, sample_rate: int = Config.SAMPLE_RATE):
        self.sr = sample_rate
        if HAS_PEDALBOARD:
            self._board = Pedalboard([PitchShift(semitones=0.0)])

    # ── pitch ─────────────────────────────────────────────────────────────────
    def pitch_shift(self, audio: np.ndarray, semitones: float) -> np.ndarray:
        if abs(semitones) < 0.5:
            return audio

        if HAS_PEDALBOARD:
            self._board[0].semitones = float(semitones)
            return self._board(audio, self.sr)

        # Resampling fallback
        ratio      = 2.0 ** (semitones / 12.0)
        orig_len   = len(audio)
        resampled  = librosa.resample(
            audio,
            orig_sr = int(self.sr * ratio),
            target_sr = self.sr,
            res_type  = 'kaiser_fast'
        )
        return self._fit(resampled, orig_len)

    # ── speed ─────────────────────────────────────────────────────────────────
    def time_stretch(self, audio: np.ndarray, rate: float) -> np.ndarray:
        """Speed change via resampling — fast, no heavy phase vocoder."""
        if abs(rate - 1.0) < 0.05:
            return audio
        orig_len  = len(audio)
        stretched = librosa.resample(
            audio,
            orig_sr   = int(self.sr * rate),
            target_sr = self.sr,
            res_type  = 'kaiser_fast'
        )
        return self._fit(stretched, orig_len)

    # ── smoothing ─────────────────────────────────────────────────────────────
    def crossfade(
        self,
        prev : np.ndarray,
        curr : np.ndarray,
        fade_len : int = Config.CROSSFADE_LEN
    ) -> np.ndarray:
        """
        Blend tail of previous chunk into head of current chunk.
        Eliminates the click that occurs when two chunks meet at
        different amplitude levels.
        """
        fade_len = min(fade_len, len(prev), len(curr))
        out      = curr.copy()
        fade_out = np.linspace(1.0, 0.0, fade_len, dtype=np.float32)
        fade_in  = np.linspace(0.0, 1.0, fade_len, dtype=np.float32)
        out[:fade_len] = (
            prev[-fade_len:] * fade_out +
            curr[:fade_len]  * fade_in
        )
        return out

    # ── utility ───────────────────────────────────────────────────────────────
    @staticmethod
    def normalize(audio: np.ndarray, ceiling: float = 0.9) -> np.ndarray:
        peak = np.max(np.abs(audio))
        return audio / peak * ceiling if peak > 1e-6 else audio

    @staticmethod
    def _fit(audio: np.ndarray, target_len: int) -> np.ndarray:
        """Trim or zero-pad to exactly target_len samples."""
        if len(audio) >= target_len:
            return audio[:target_len]
        return np.pad(audio, (0, target_len - len(audio)))


# =============================================================================
# AUDIO ENGINE  —  FILE mode + MIC mode
# =============================================================================

class AudioEngine:
    """
    Manages audio I/O and the processing pipeline.

    Threading model
    ───────────────
    FILE mode:
        _process_loop  (daemon)  reads file chunks, applies DSP, stores result
        _play_loop     (daemon)  reads stored result, writes to output device

    MIC mode:
        sounddevice InputStream  calls _mic_callback on every input block
        _mic_callback             applies DSP in-place, stores result
        _play_loop     (daemon)  reads stored result, writes to output device

    The double-buffer (_output_buf + _buf_lock) decouples production speed
    from consumption speed.  The play thread never starves — worst case it
    replays the previous chunk (brief repeat is less noticeable than silence).
    """

    def __init__(
        self,
        mode          : AudioMode,
        gesture_state : GestureState,
        audio_file    : str = None
    ):
        self.mode    = mode
        self.gesture = gesture_state
        self.dsp     = DSP()
        self.running = False

        # Double buffer
        self._output_buf  = np.zeros(Config.CHUNK_SIZE, dtype=np.float32)
        self._buf_lock    = threading.Lock()
        self._prev_chunk  = np.zeros(Config.CHUNK_SIZE, dtype=np.float32)

        if mode == AudioMode.FILE:
            if not audio_file:
                raise ValueError("audio_file path required in FILE mode")
            self._load_file(audio_file)
            self._pos = 0

    # ── file loading ──────────────────────────────────────────────────────────
    def _load_file(self, path: str) -> None:
        print(f"📂 Loading {path}...")
        audio, _ = librosa.load(path, sr=Config.SAMPLE_RATE, mono=True)
        self._audio = audio.astype(np.float32)
        peak = np.max(np.abs(self._audio))
        if peak > 0:
            self._audio /= peak
        duration = len(self._audio) / Config.SAMPLE_RATE
        print(f"✅ Loaded — {duration:.1f} seconds")

    def _next_file_chunk(self) -> np.ndarray:
        """Extract next CHUNK_SIZE samples, looping at end."""
        end = self._pos + Config.CHUNK_SIZE
        if end >= len(self._audio):
            wrap  = end - len(self._audio)
            chunk = np.concatenate([self._audio[self._pos:], self._audio[:wrap]])
            self._pos = wrap
        else:
            chunk     = self._audio[self._pos:end].copy()
            self._pos = end
        return chunk

    # ── DSP chain (shared by both modes) ─────────────────────────────────────
    def _process_chunk(self, chunk: np.ndarray) -> np.ndarray:
        g = self.gesture.get()

        chunk = self.dsp.pitch_shift(chunk, g["pitch"])
        chunk = self.dsp.time_stretch(chunk, g["speed"])
        chunk = DSP._fit(chunk, Config.CHUNK_SIZE)
        chunk = chunk * g["volume"]
        chunk = self.dsp.normalize(chunk)
        chunk = self.dsp.crossfade(self._prev_chunk, chunk)
        self._prev_chunk = chunk.copy()

        return chunk

    # ── FILE mode thread ──────────────────────────────────────────────────────
    def _process_loop(self) -> None:
        while self.running:
            try:
                raw       = self._next_file_chunk()
                processed = self._process_chunk(raw)
                with self._buf_lock:
                    self._output_buf = processed
            except Exception as exc:
                print(f"[AudioEngine] process error: {exc}")

    # ── MIC mode callback (called by sounddevice, not a thread) ──────────────
    def _mic_callback(self, indata, frames, time_info, status) -> None:
        chunk = indata[:, 0].copy()
        if np.max(np.abs(chunk)) < 0.005:
            return   # skip near-silence to reduce CPU
        processed = self._process_chunk(chunk)
        with self._buf_lock:
            self._output_buf = processed

    # ── play thread (shared by both modes) ───────────────────────────────────
    def _play_loop(self) -> None:
        with sd.OutputStream(
            samplerate = Config.SAMPLE_RATE,
            channels   = 1,
            dtype      = 'float32',
            device     = Config.OUTPUT_DEVICE,
            blocksize  = Config.CHUNK_SIZE
        ) as stream:
            while self.running:
                try:
                    with self._buf_lock:
                        chunk = self._output_buf.copy()
                    stream.write(chunk.reshape(-1, 1))
                except Exception:
                    # Write silence rather than crashing the stream
                    silence = np.zeros((Config.CHUNK_SIZE, 1), dtype=np.float32)
                    stream.write(silence)

    # ── lifecycle ─────────────────────────────────────────────────────────────
    def start(self) -> None:
        self.running = True

        if self.mode == AudioMode.FILE:
            threading.Thread(
                target=self._process_loop, daemon=True, name="ProcessLoop"
            ).start()

        elif self.mode == AudioMode.MIC:
            self._mic_stream = sd.InputStream(
                samplerate = Config.SAMPLE_RATE,
                channels   = 1,
                blocksize  = Config.CHUNK_SIZE,
                dtype      = 'float32',
                device     = Config.INPUT_DEVICE,
                callback   = self._mic_callback
            )
            self._mic_stream.start()

        threading.Thread(
            target=self._play_loop, daemon=True, name="PlayLoop"
        ).start()

    def stop(self) -> None:
        self.running = False
        if self.mode == AudioMode.MIC and hasattr(self, '_mic_stream'):
            self._mic_stream.stop()
        time.sleep(0.4)
        sd.stop()


# =============================================================================
# GESTURE CONTROLLER  —  webcam + MediaPipe
# =============================================================================

class GestureController:
    """
    Reads webcam frames, runs MediaPipe hand tracking,
    maps landmark positions to audio parameters.

    Landmark index reference:
        4  = thumb tip
        8  = index finger tip
        12 = middle finger tip
    """

    _FINGER_LABELS = {
        8:  ("PITCH",  (0, 255,   0)),
        4:  ("SPEED",  (255, 0,   0)),
        12: ("VOL",    (0,   0, 255)),
    }

    def __init__(self, gesture_state: GestureState):
        self.gesture = gesture_state

        mp_hands        = mp.solutions.hands
        self._mp_draw   = mp.solutions.drawing_utils
        self._mp_hands  = mp_hands
        self._detector  = mp_hands.Hands(
            static_image_mode        = False,
            max_num_hands            = 1,
            min_detection_confidence = 0.7,
            min_tracking_confidence  = 0.7
        )

        self._cap = cv2.VideoCapture(0)
        if not self._cap.isOpened():
            raise RuntimeError("❌ Could not open webcam")

    # ── gesture mapping ───────────────────────────────────────────────────────
    def _update_gestures(self, landmarks, h: int, w: int) -> dict[int, tuple]:
        """Map finger positions → audio parameters, return pixel coords."""
        idx = landmarks.landmark[8]
        thb = landmarks.landmark[4]
        mid = landmarks.landmark[12]

        self.gesture.update(
            pitch  = np.interp(idx.y, [0.1, 0.9],
                               [Config.PITCH_RANGE[1],  Config.PITCH_RANGE[0]]),
            speed  = np.interp(thb.x, [0.1, 0.9],
                               [Config.SPEED_RANGE[0],  Config.SPEED_RANGE[1]]),
            volume = np.interp(mid.y, [0.1, 0.9],
                               [Config.VOLUME_RANGE[1], Config.VOLUME_RANGE[0]]),
        )

        return {
            8:  (int(idx.x * w), int(idx.y * h)),
            4:  (int(thb.x * w), int(thb.y * h)),
            12: (int(mid.x * w), int(mid.y * h)),
        }

    # ── overlay drawing ───────────────────────────────────────────────────────
    @staticmethod
    def _draw_hud(frame: np.ndarray, g: dict, mode: AudioMode) -> None:
        cv2.rectangle(frame, (0, 0), (290, 110), (0, 0, 0), -1)
        cv2.putText(frame, f"PITCH : {g['pitch']:+.1f} st",
                    (10, 30),  cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 0), 2)
        cv2.putText(frame, f"SPEED : {g['speed']:.2f}x",
                    (10, 60),  cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 0, 0), 2)
        cv2.putText(frame, f"VOL   : {g['volume']:.2f}",
                    (10, 90),  cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 255), 2)

        label = f"[{mode.value.upper()}]"
        cv2.putText(frame, label,
                    (frame.shape[1] - 90, 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

    # ── main read ─────────────────────────────────────────────────────────────
    def read_frame(self, mode: AudioMode):
        """Returns annotated frame, or None on camera failure."""
        ret, frame = self._cap.read()
        if not ret:
            return None

        frame    = cv2.flip(frame, 1)
        h, w, _  = frame.shape
        rgb      = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results  = self._detector.process(rgb)

        if results.multi_hand_landmarks:
            for hand_lm in results.multi_hand_landmarks:
                self._mp_draw.draw_landmarks(
                    frame, hand_lm, self._mp_hands.HAND_CONNECTIONS
                )
                coords = self._update_gestures(hand_lm, h, w)

                for idx, (pt) in coords.items():
                    label, color = self._FINGER_LABELS[idx]
                    cv2.circle(frame, pt, 10, color, -1)
                    cv2.putText(frame, label, (pt[0] + 12, pt[1]),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        self._draw_hud(frame, self.gesture.get(), mode)
        return frame

    def release(self) -> None:
        self._cap.release()
        self._detector.close()


# =============================================================================
# MAIN ORCHESTRATOR
# =============================================================================

class GestureAudioModulator:
    """
    Top-level class.  Owns the engine and controller, runs the event loop.
    """

    def __init__(self, mode: AudioMode, audio_file: str = None):
        self.mode    = mode
        self.gesture = GestureState()
        self.engine  = AudioEngine(mode, self.gesture, audio_file)
        self.camera  = GestureController(self.gesture)

    def run(self) -> None:
        print(f"\n🎵 GBAM-v5 — mode: {self.mode.value.upper()}")
        print("   Index ↑↓ = Pitch | Thumb ←→ = Speed | Middle ↑↓ = Volume")
        print("   Press Q to quit\n")

        self.engine.start()

        try:
            while True:
                frame = self.camera.read_frame(self.mode)
                if frame is None:
                    print("Camera read failed — exiting")
                    break

                cv2.imshow("GBAM — Gesture Audio Modulator", frame)

                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

        except KeyboardInterrupt:
            print("\n⛔ Interrupted")
        finally:
            print("Shutting down...")
            self.engine.stop()
            self.camera.release()
            cv2.destroyAllWindows()
            print("✅ Done")


# =============================================================================
# ENTRY POINT
# =============================================================================

if __name__ == "__main__":
    """
    Usage:
        python GBAM-v5.py file Bheegi.mp3   — song file mode
        python GBAM-v5.py file              — song file mode (default: Bheegi.mp3)
        python GBAM-v5.py mic               — live mic mode
        python GBAM-v5.py                   — defaults to file mode
    """

    args = sys.argv[1:]

    if args and args[0] == "mic":
        app = GestureAudioModulator(mode=AudioMode.MIC)
    else:
        audio_file = args[1] if len(args) > 1 else "Bheegi.mp3"
        app = GestureAudioModulator(mode=AudioMode.FILE, audio_file=audio_file)

    app.run()
