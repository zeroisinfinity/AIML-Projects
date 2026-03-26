import os
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GLOG_minloglevel'] = '2'
os.environ['QT_LOGGING_RULES'] = '*.debug=false'
os.environ['QT_QPA_PLATFORM']  = 'xcb'

import cv2
import mediapipe as mp
import numpy as np
import sounddevice as sd
import librosa
import scipy.io.wavfile as wav
import threading
import warnings
warnings.filterwarnings("ignore")

# =====================
# LOAD VOICE SAMPLE
# =====================
print("📂 Loading voice_sample.wav...")
sr, audio_raw = wav.read("voice_sample.wav")

if audio_raw.dtype != np.float32:
    audio_raw = audio_raw.astype(np.float32)
    if np.max(np.abs(audio_raw)) > 1.0:
        audio_raw = audio_raw / 32768.0

if len(audio_raw.shape) > 1:
    audio_raw = audio_raw[:, 0]

if sr != 22050:
    audio_raw = librosa.resample(audio_raw, orig_sr=sr, target_sr=22050)

SAMPLE_RATE   = 22050
OUTPUT_DEVICE = 13
CHUNK_SIZE    = 16384   # larger = smoother
CROSSFADE     = 2048    # samples to blend between chunks

print(f"✅ Loaded — {len(audio_raw)/SAMPLE_RATE:.1f} seconds")

# =====================
# MEDIAPIPE
# =====================
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

# =====================
# WEBCAM
# =====================
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("❌ Webcam not found")
    exit()

# =====================
# SHARED STATE
# =====================
gesture = {
    "pitch": 0.0,
    "speed": 1.0
}

latest_output = [np.zeros(CHUNK_SIZE, dtype=np.float32)]
latest_lock   = threading.Lock()
running       = [True]

# =====================
# CROSSFADE FUNCTION
# blends end of previous chunk into start of new chunk
# eliminates clicks between chunks
# =====================
def crossfade(prev, curr, fade_len):
    # fade out = 1.0 → 0.0 over fade_len samples
    fade_out = np.linspace(1.0, 0.0, fade_len).astype(np.float32)
    # fade in  = 0.0 → 1.0 over fade_len samples
    fade_in  = np.linspace(0.0, 1.0, fade_len).astype(np.float32)

    result = curr.copy()

    # Blend: end of prev * fade_out + start of curr * fade_in
    result[:fade_len] = (prev[-fade_len:] * fade_out +
                         curr[:fade_len]  * fade_in)
    return result

# =====================
# PLAYBACK THREAD
# =====================
def playback_loop():
    pos       = 0
    prev_chunk = np.zeros(CHUNK_SIZE, dtype=np.float32)

    while running[0]:
        try:
            # Grab chunk from audio file
            end   = pos + CHUNK_SIZE
            chunk = audio_raw[pos:end].copy()

            # Loop back if near end
            if len(chunk) < CHUNK_SIZE:
                remaining = CHUNK_SIZE - len(chunk)
                chunk     = np.concatenate([chunk, audio_raw[:remaining]])
                pos       = remaining
            else:
                pos = end

            # Apply pitch shift
            pitch_steps = gesture["pitch"]
            speed_rate  = gesture["speed"]

            if abs(pitch_steps) > 0.5:
                chunk = librosa.effects.pitch_shift(
                    y=chunk,
                    sr=SAMPLE_RATE,
                    n_steps=float(pitch_steps)
                )

            if abs(speed_rate - 1.0) > 0.1:
                chunk = librosa.effects.time_stretch(
                    y=chunk,
                    rate=float(speed_rate)
                )

            # Pad or trim
            if len(chunk) < CHUNK_SIZE:
                chunk = np.pad(chunk, (0, CHUNK_SIZE - len(chunk)))
            else:
                chunk = chunk[:CHUNK_SIZE]

            # Normalize
            max_val = np.max(np.abs(chunk))
            if max_val > 0:
                chunk = chunk / max_val * 0.9

            # Crossfade with previous chunk
            chunk = crossfade(prev_chunk, chunk, CROSSFADE)
            prev_chunk = chunk.copy()

            with latest_lock:
                latest_output[0] = chunk

        except Exception as e:
            print(f"Playback error: {e}")
            continue

# =====================
# PLAY THREAD
# =====================
def play_audio():
    with sd.OutputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype='float32',
        device=OUTPUT_DEVICE,
        blocksize=CHUNK_SIZE
    ) as out_stream:
        while running[0]:
            try:
                with latest_lock:
                    chunk = latest_output[0].copy()
                out_stream.write(chunk.reshape(-1, 1))
            except Exception:
                silence = np.zeros((CHUNK_SIZE, 1), dtype=np.float32)
                out_stream.write(silence)

# =====================
# START THREADS
# =====================
t_playback = threading.Thread(target=playback_loop, daemon=True)
t_play     = threading.Thread(target=play_audio,    daemon=True)

t_playback.start()
t_play.start()

print("✅ Starting — index finger UP/DOWN = pitch | thumb LEFT/RIGHT = speed")
print("✅ Press Q to quit")

# =====================
# MAIN VIDEO LOOP
# =====================
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame     = cv2.flip(frame, 1)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results   = hands.process(rgb_frame)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:

                mp_draw.draw_landmarks(
                    frame,
                    hand_landmarks,
                    mp_hands.HAND_CONNECTIONS
                )

                h, w, _ = frame.shape

                index_tip = hand_landmarks.landmark[8]
                cx_index  = int(index_tip.x * w)
                cy_index  = int(index_tip.y * h)

                thumb_tip = hand_landmarks.landmark[4]
                cx_thumb  = int(thumb_tip.x * w)
                cy_thumb  = int(thumb_tip.y * h)

                gesture["pitch"] = np.interp(cy_index, [0, h], [12.0, -12.0])
                gesture["speed"] = np.interp(cx_thumb, [0, w], [0.5, 2.0])

                cv2.circle(frame, (cx_index, cy_index), 12, (0, 255, 0), -1)
                cv2.circle(frame, (cx_thumb, cy_thumb), 12, (255, 0, 0), -1)

                cv2.putText(frame, "PITCH", (cx_index + 15, cy_index),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                cv2.putText(frame, "SPEED", (cx_thumb + 15, cy_thumb),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 0), 2)

        # HUD
        cv2.rectangle(frame, (0, 0), (300, 90), (0, 0, 0), -1)
        cv2.putText(frame, f"PITCH : {gesture['pitch']:+.1f} semitones",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"SPEED : {gesture['speed']:.2f}x",
                    (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        cv2.imshow("Gesture Voice Control", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    print("\n⛔ Stopping...")

finally:
    running[0] = False
    import time
    time.sleep(0.5)     # give threads time to see running=False and stop
    sd.stop()           # force stop all sounddevice streams
    cap.release()
    cv2.destroyAllWindows()
    print("✅ Cleaned up")