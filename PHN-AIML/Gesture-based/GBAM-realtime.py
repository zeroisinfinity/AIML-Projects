import os
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GLOG_minloglevel'] = '2'
os.environ['QT_LOGGING_RULES'] = '*.debug=false'
os.environ['PYTHONWARNINGS'] = 'ignore'

import cv2
import mediapipe as mp
import numpy as np
import sounddevice as sd
import librosa
import queue
import threading
import warnings
warnings.filterwarnings("ignore")

# =====================
# AUDIO CONFIG
# =====================
INPUT_DEVICE  = 13
OUTPUT_DEVICE = 13
SAMPLE_RATE   = 22050
CHUNK_SIZE    = 8192        # larger = more stable, less underrun
SILENCE       = np.zeros(8192, dtype=np.float32)

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

audio_queue  = queue.Queue(maxsize=5)
output_queue = queue.Queue(maxsize=5)

# Latest processed chunk — output thread always has something to play
latest_output = [SILENCE.copy()]
latest_lock   = threading.Lock()

# =====================
# MIC CALLBACK
# =====================
def audio_callback(indata, frames, time, status):
    if not audio_queue.full():
        audio_queue.put(indata.copy())

# =====================
# PROCESS THREAD
# =====================
def process_audio():
    while True:
        try:
            raw        = audio_queue.get(timeout=1)
            audio_data = raw[:, 0].astype(np.float32)

            # Skip silence
            if np.max(np.abs(audio_data)) < 0.01:
                with latest_lock:
                    latest_output[0] = audio_data
                continue

            pitch_steps = gesture["pitch"]
            speed_rate  = gesture["speed"]

            # Only pitch shift if gesture is far enough from center
            if abs(pitch_steps) > 0.5:
                audio_data = librosa.effects.pitch_shift(
                    y=audio_data,
                    sr=SAMPLE_RATE,
                    n_steps=float(pitch_steps)
                )

            # Only time stretch if speed is far enough from 1.0
            if abs(speed_rate - 1.0) > 0.1:
                audio_data = librosa.effects.time_stretch(
                    y=audio_data,
                    rate=float(speed_rate)
                )

            # Normalize to prevent clipping
            max_val = np.max(np.abs(audio_data))
            if max_val > 0:
                audio_data = audio_data / max_val * 0.9

            # Store latest processed chunk
            with latest_lock:
                latest_output[0] = audio_data

        except queue.Empty:
            continue
        except Exception as e:
            print(f"Process error: {e}")
            continue

# =====================
# PLAY THREAD
# uses latest_output — never starves, always has something
# =====================
def play_audio():
    with sd.OutputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype='float32',
        device=OUTPUT_DEVICE,
        blocksize=CHUNK_SIZE
    ) as out_stream:
        while True:
            try:
                with latest_lock:
                    chunk = latest_output[0].copy()

                # Pad or trim to exact CHUNK_SIZE
                if len(chunk) < CHUNK_SIZE:
                    chunk = np.pad(chunk, (0, CHUNK_SIZE - len(chunk)))
                elif len(chunk) > CHUNK_SIZE:
                    chunk = chunk[:CHUNK_SIZE]

                out_stream.write(chunk.reshape(-1, 1))

            except Exception as e:
                print(f"Play error: {e}")
                # On error write silence — never let stream starve
                out_stream.write(SILENCE.reshape(-1, 1))

# =====================
# START THREADS
# =====================
t_process = threading.Thread(target=process_audio, daemon=True)
t_play    = threading.Thread(target=play_audio,    daemon=True)

t_process.start()
t_play.start()

print("✅ Starting — Press Q to quit")
print(f"🎤 Input  : {sd.query_devices(INPUT_DEVICE)['name']}")
print(f"🔊 Output : {sd.query_devices(OUTPUT_DEVICE)['name']}")

# =====================
# MAIN VIDEO LOOP
# =====================
with sd.InputStream(
    samplerate=SAMPLE_RATE,
    channels=1,
    blocksize=CHUNK_SIZE,
    dtype='float32',
    device=INPUT_DEVICE,
    callback=audio_callback
):
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

        cv2.rectangle(frame, (0, 0), (300, 90), (0, 0, 0), -1)
        cv2.putText(frame, f"PITCH : {gesture['pitch']:+.1f} semitones",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, f"SPEED : {gesture['speed']:.2f}x",
                    (10, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)

        cv2.imshow("Gesture Voice Control", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
print("✅ Done")
