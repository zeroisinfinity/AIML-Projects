import cv2
import mediapipe as mp
import numpy as np
import sounddevice as sd
import librosa
import scipy.signal as signal
from collections import deque
import threading
import queue
import time

cap = cv2.VideoCapture(0)  # 0 = first webcam

if cap.isOpened():
    print("✅ Webcam opened successfully")
    ret, frame = cap.read()  # read ONE frame
    if ret:
        print(f"✅ Frame captured successfully")
        print(f"Frame shape: {frame.shape}")  # should print (height, width, 3)
    else:
        print("❌ Could not read frame")
else:
    print("❌ Could not open webcam")

cap.release()  # ALWAYS release after done
print("Webcam released ✅")

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

cap = cv2.VideoCapture(0)
ret, frame = cap.read()
cap.release()

rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
results = hands.process(rgb_frame)

if results.multi_hand_landmarks:
    print("✅ Hand detected!")
    landmarks = results.multi_hand_landmarks[0]
    for idx, lm in enumerate(landmarks.landmark):
        print(f"Point {idx}: x={lm.x:.3f}, y={lm.y:.3f}, z={lm.z:.3f}")
else:
    print("❌ No hand detected — make sure hand is visible to camera")