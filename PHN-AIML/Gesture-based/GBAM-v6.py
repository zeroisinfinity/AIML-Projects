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

# -------- load audio --------
print("loading audio...")
sr, audio_raw = wav.read("Sooraj.wav")

# basic cleanup
if audio_raw.dtype != np.float32:
    audio_raw = audio_raw.astype(np.float32)
    if np.max(np.abs(audio_raw)) > 1.0:
        audio_raw /= 32768.0

# mono only
if len(audio_raw.shape) > 1:
    audio_raw = audio_raw[:, 0]

# force sample rate
if sr != 22050:
    audio_raw = librosa.resample(audio_raw, orig_sr=sr, target_sr=22050)

SAMPLE_RATE   = 22050
OUTPUT_DEVICE = 13
CHUNK_SIZE    = 102400   # small chunk = responsive
CROSSFADE     = 25600    # just enough to hide clicks

print("ready")

# -------- mediapipe --------
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)
# -------- camera --------
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    print("camera not working")
    exit()

# -------- shared state --------
gesture = {"pitch": 0.0, "speed": 1.0}

latest_output = [np.zeros(CHUNK_SIZE, dtype=np.float32)]
lock = threading.Lock()
running = [True]

# -------- crossfade --------
def crossfade(prev, curr, n):
    n = min(n, len(prev), len(curr))

    fade_out = np.linspace(1.0, 0.0, n).astype(np.float32)
    fade_in  = np.linspace(0.0, 1.0, n).astype(np.float32)

    out = curr.copy()
    out[:n] = prev[-n:] * fade_out + curr[:n] * fade_in
    return out

# -------- audio worker --------
def playback_loop():
    pos = 0
    prev = np.zeros(CHUNK_SIZE, dtype=np.float32)

    while running[0]:
        try:
            chunk = audio_raw[pos:pos + CHUNK_SIZE].copy()

            if len(chunk) < CHUNK_SIZE:
                chunk = np.concatenate([chunk, audio_raw[:CHUNK_SIZE - len(chunk)]])
                pos = 0
            else:
                pos += CHUNK_SIZE

            # simple controls
            pitch = gesture["pitch"]
            speed = gesture["speed"]

            if abs(pitch) > 0.5:
                chunk = librosa.effects.pitch_shift(chunk, SAMPLE_RATE, pitch)

            if abs(speed - 1.0) > 0.1:
                chunk = librosa.effects.time_stretch(chunk, speed)

            # fix length
            if len(chunk) < CHUNK_SIZE:
                chunk = np.pad(chunk, (0, CHUNK_SIZE - len(chunk)))
            else:
                chunk = chunk[:CHUNK_SIZE]

            # normalize a bit
            m = np.max(np.abs(chunk))
            if m > 0:
                chunk = chunk / m * 0.9

            # smooth edges
            chunk = crossfade(prev, chunk, CROSSFADE)
            prev = chunk.copy()

            with lock:
                latest_output[0] = chunk

        except:
            pass

# -------- audio output --------
def play_audio():
    with sd.OutputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype='float32',
        device=OUTPUT_DEVICE,
        blocksize=CHUNK_SIZE
    ) as stream:
        while running[0]:
            try:
                with lock:
                    chunk = latest_output[0].copy()
                stream.write(chunk.reshape(-1, 1))
            except:
                stream.write(np.zeros((CHUNK_SIZE, 1), dtype=np.float32))

# -------- start threads --------
threading.Thread(target=playback_loop, daemon=True).start()
threading.Thread(target=play_audio, daemon=True).start()

print("use index finger (up/down) for pitch, thumb (left/right) for speed")
print("press q to quit")

# -------- main loop --------
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb   = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res   = hands.process(rgb)

        if res.multi_hand_landmarks:
            for hand in res.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, hand, mp_hands.HAND_CONNECTIONS)

                h, w, _ = frame.shape

                idx = hand.landmark[8]
                th  = hand.landmark[4]

                x_i, y_i = int(idx.x * w), int(idx.y * h)
                x_t, y_t = int(th.x * w), int(th.y * h)

                pitch_raw = np.interp(y_i, [0, h], [12, -12])
                speed_raw = np.interp(x_t, [0, w], [0.5, 2.0])

                alpha = 0.15  # smoothing strength

                gesture["pitch"] += alpha * (pitch_raw - gesture["pitch"])
                gesture["speed"] += alpha * (speed_raw - gesture["speed"])

                cv2.circle(frame, (x_i, y_i), 10, (0,255,0), -1)
                cv2.circle(frame, (x_t, y_t), 10, (255,0,0), -1)

        # simple HUD
        cv2.putText(frame, f"pitch: {gesture['pitch']:.1f}",
                    (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,255,0), 2)
        cv2.putText(frame, f"speed: {gesture['speed']:.2f}",
                    (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,0,0), 2)

        cv2.imshow("gesture audio demo", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

except KeyboardInterrupt:
    pass

finally:
    running[0] = False
    import time
    time.sleep(0.3)
    sd.stop()
    cap.release()
    cv2.destroyAllWindows()