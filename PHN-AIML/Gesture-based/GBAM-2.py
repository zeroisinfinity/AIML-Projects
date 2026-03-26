# file: final_smooth_mp3.py

import os
os.environ['GRPC_VERBOSITY'] = 'ERROR'
os.environ['GLOG_minloglevel'] = '2'

import cv2
import mediapipe as mp
import numpy as np
import sounddevice as sd
import librosa
import threading

# =====================
# LOAD AUDIO
# =====================
audio_raw, sr = librosa.load("Bheegi.mp3", sr=22050)
audio_raw = audio_raw.astype(np.float32)

SAMPLE_RATE = 22050
OUTPUT_DEVICE = 13

# 🔥 BALANCED CONFIG
CHUNK_SIZE   = 4096
CROSSFADE    = 512
OUTPUT_BLOCK = 512
OVERLAP      = 512

# =====================
# MEDIAPIPE
# =====================
mp_hands = mp.solutions.hands
mp_draw  = mp.solutions.drawing_utils
hands = mp_hands.Hands(max_num_hands=1)

cap = cv2.VideoCapture(0)

# =====================
# STATE
# =====================
gesture = {
    "pitch": 0.0,
    "speed": 1.0,
    "volume": 1.0
}

latest_output = [np.zeros(CHUNK_SIZE, dtype=np.float32)]
latest_lock = threading.Lock()
running = [True]

# =====================
# CROSSFADE
# =====================
def crossfade(prev, curr):
    fade_out = np.linspace(1, 0, CROSSFADE)
    fade_in  = np.linspace(0, 1, CROSSFADE)
    curr[:CROSSFADE] = prev[-CROSSFADE:] * fade_out + curr[:CROSSFADE] * fade_in
    return curr

# =====================
# PLAYBACK LOOP (SMOOTH)
# =====================
def playback_loop():
    pos = 0
    prev_chunk = np.zeros(CHUNK_SIZE, dtype=np.float32)
    prev_tail  = np.zeros(OVERLAP, dtype=np.float32)

    while running[0]:
        end = pos + CHUNK_SIZE
        chunk = audio_raw[pos:end]

        if len(chunk) < CHUNK_SIZE:
            chunk = np.concatenate([chunk, audio_raw[:CHUNK_SIZE - len(chunk)]])
            pos = CHUNK_SIZE - len(chunk)
        else:
            pos = end % len(audio_raw)

        # 🔥 OVERLAP ADD
        chunk = np.concatenate([prev_tail, chunk])

        # DSP (librosa)
        if abs(gesture["pitch"]) > 0.5:
            chunk = librosa.effects.pitch_shift(chunk, SAMPLE_RATE, gesture["pitch"])

        if abs(gesture["speed"] - 1.0) > 0.1:
            chunk = librosa.effects.time_stretch(chunk, gesture["speed"])

        # 🔥 SAVE OVERLAP
        prev_tail = chunk[-OVERLAP:]
        chunk = chunk[:-OVERLAP]

        # volume
        chunk *= gesture["volume"]

        # normalize
        m = np.max(np.abs(chunk))
        if m > 0:
            chunk = chunk / m * 0.9

        # crossfade
        chunk = crossfade(prev_chunk, chunk)
        prev_chunk = chunk.copy()

        with latest_lock:
            latest_output[0] = chunk

# =====================
# AUDIO OUTPUT
# =====================
def play_audio():
    with sd.OutputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype='float32',
        device=OUTPUT_DEVICE,
        blocksize=OUTPUT_BLOCK
    ) as stream:

        while running[0]:
            with latest_lock:
                chunk = latest_output[0].copy()

            for i in range(0, len(chunk), OUTPUT_BLOCK):
                small = chunk[i:i+OUTPUT_BLOCK]

                if len(small) < OUTPUT_BLOCK:
                    small = np.pad(small, (0, OUTPUT_BLOCK - len(small)))

                stream.write(small.reshape(-1,1))

# =====================
# START THREADS
# =====================
threading.Thread(target=playback_loop, daemon=True).start()
threading.Thread(target=play_audio, daemon=True).start()

print("Running...")

# =====================
# MAIN LOOP
# =====================
try:
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frame = cv2.flip(frame, 1)
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        res = hands.process(rgb)

        if res.multi_hand_landmarks:
            for lm in res.multi_hand_landmarks:
                mp_draw.draw_landmarks(frame, lm, mp_hands.HAND_CONNECTIONS)

                h,w,_ = frame.shape

                i = lm.landmark[8]
                t = lm.landmark[4]
                m = lm.landmark[12]

                cx_i, cy_i = int(i.x*w), int(i.y*h)
                cx_t, cy_t = int(t.x*w), int(t.y*h)
                cx_m, cy_m = int(m.x*w), int(m.y*h)

                gesture["pitch"]  = np.interp(cy_i,[0,h],[12,-12])
                gesture["speed"]  = np.interp(cx_t,[0,w],[0.5,2.0])
                gesture["volume"] = np.interp(cy_m,[0,h],[1.5,0.2])

                cv2.circle(frame,(cx_i,cy_i),10,(0,255,0),-1)
                cv2.circle(frame,(cx_t,cy_t),10,(255,0,0),-1)
                cv2.circle(frame,(cx_m,cy_m),10,(0,0,255),-1)

        cv2.imshow("Gesture Voice", frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

finally:
    running[0] = False
    cap.release()
    cv2.destroyAllWindows()