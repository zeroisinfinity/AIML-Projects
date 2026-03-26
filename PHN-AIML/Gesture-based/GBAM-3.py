import os

os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GLOG_minloglevel'] = '2'
os.environ['QT_LOGGING_RULES'] = '*.debug=false'
os.environ['QT_QPA_PLATFORM'] = 'xcb'

import cv2
import mediapipe as mp
import numpy as np
import sounddevice as sd
import librosa
import threading
import warnings

warnings.filterwarnings("ignore")

# =====================
# CONFIGURATION
# =====================
SAMPLE_RATE = 22050
OUTPUT_DEVICE = 13
CHUNK_SIZE = 8192  # Optimized for real-time processing
CROSSFADE = 2048  # Smooths chunk transitions
FILE_NAME = "Bheegi.mp3"

# =====================
# LOAD AUDIO
# =====================
print(f"📂 Loading {FILE_NAME}...")
audio_raw, sr = librosa.load(FILE_NAME, sr=SAMPLE_RATE)
audio_raw = audio_raw.astype(np.float32)

# Normalize
if np.max(np.abs(audio_raw)) > 0:
    audio_raw = audio_raw / np.max(np.abs(audio_raw))

print(f"✅ Loaded — {len(audio_raw) / SAMPLE_RATE:.1f} seconds")

# =====================
# MEDIAPIPE & WEBCAM
# =====================
mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1, min_detection_confidence=0.7, min_tracking_confidence=0.7)

cap = cv2.VideoCapture(0)

# =====================
# SHARED STATE
# =====================
gesture = {"pitch": 0.0, "speed": 1.0, "volume": 1.0}
latest_output = [np.zeros(CHUNK_SIZE, dtype=np.float32)]
latest_lock = threading.Lock()
running = [True]


def crossfade(prev, curr, fade_len):
    fade_out = np.linspace(1.0, 0.0, fade_len).astype(np.float32)
    fade_in = np.linspace(0.0, 1.0, fade_len).astype(np.float32)
    result = curr.copy()
    result[:fade_len] = (prev[-fade_len:] * fade_out + curr[:fade_len] * fade_in)
    return result


# =====================
# PLAYBACK THREAD (The Engine)
# =====================
def playback_loop():
    pos_float = 0.0
    prev_chunk = np.zeros(CHUNK_SIZE, dtype=np.float32)

    while running[0]:
        try:
            # 1. Calculate read size based on dynamic speed
            read_amount = int(CHUNK_SIZE * gesture["speed"])
            start = int(pos_float)
            end = start + read_amount

            # 2. Extract and Wrap-around
            if end >= len(audio_raw):
                chunk = np.concatenate([audio_raw[start:], audio_raw[:end - len(audio_raw)]])
                pos_float = float(end - len(audio_raw))
            else:
                chunk = audio_raw[start:end].copy()
                pos_float += read_amount

            # 3. Processing: Speed via Resampling (Faster than time_stretch)
            if len(chunk) != CHUNK_SIZE:
                chunk = librosa.resample(chunk, orig_sr=len(chunk), target_sr=CHUNK_SIZE)

            # 4. Processing: Pitch (Optimized)
            if abs(gesture["pitch"]) > 0.5:
                chunk = librosa.effects.pitch_shift(
                    y=chunk, sr=SAMPLE_RATE,
                    n_steps=float(gesture["pitch"]),
                    res_type='kaiser_fast'
                )

            # 5. Volume & Smoothing
            chunk = chunk * gesture["volume"]
            chunk = crossfade(prev_chunk, chunk, CROSSFADE)
            prev_chunk = chunk.copy()

            with latest_lock:
                latest_output[0] = chunk

        except Exception as e:
            continue


# =====================
# PLAY THREAD
# =====================
def play_audio():
    with sd.OutputStream(
            samplerate=SAMPLE_RATE, channels=1, dtype='float32',
            device=OUTPUT_DEVICE, blocksize=CHUNK_SIZE
    ) as out_stream:
        while running[0]:
            with latest_lock:
                chunk = latest_output[0].copy()
            out_stream.write(chunk.reshape(-1, 1))


# Start Threads
threading.Thread(target=playback_loop, daemon=True).start()
threading.Thread(target=play_audio, daemon=True).start()

# =====================
# MAIN VIDEO LOOP
# =====================
try:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break

        frame = cv2.flip(frame, 1)
        h, w, _ = frame.shape
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = hands.process(rgb)

        if results.multi_hand_landmarks:
            for hl in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hl, mp_hands.HAND_CONNECTIONS)

                # Finger Points
                idx = hl.landmark[8]  # Index -> Pitch
                thb = hl.landmark[4]  # Thumb -> Speed
                mid = hl.landmark[12]  # Middle -> Volume

                gesture["pitch"] = np.interp(idx.y, [0.1, 0.9], [12.0, -12.0])
                gesture["speed"] = np.interp(thb.x, [0.1, 0.9], [0.5, 2.0])
                gesture["volume"] = np.interp(mid.y, [0.1, 0.9], [1.5, 0.0])

        # HUD
        cv2.putText(frame, f"PITCH: {gesture['pitch']:+.1f}", (10, 30), 1, 1.5, (0, 255, 0), 2)
        cv2.putText(frame, f"SPEED: {gesture['speed']:.2f}x", (10, 70), 1, 1.5, (255, 0, 0), 2)

        cv2.imshow("Gesture Modulation", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'): break

finally:
    running[0] = False
    cap.release()
    cv2.destroyAllWindows()
