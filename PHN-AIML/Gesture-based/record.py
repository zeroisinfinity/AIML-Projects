import sounddevice as sd
import numpy as np
import scipy.io.wavfile as wav

SAMPLE_RATE  = 22050
DURATION     = 20      # 10 seconds
INPUT_DEVICE = 13

print("🎤 Recording in 3...")
import time
time.sleep(1)
print("2...")
time.sleep(1)
print("1... SPEAK NOW")
time.sleep(1)

recording = sd.rec(
    int(DURATION * SAMPLE_RATE),
    samplerate=SAMPLE_RATE,
    channels=1,
    dtype='float32',
    device=INPUT_DEVICE
)
sd.wait()

wav.write("voice_sample.wav", SAMPLE_RATE, recording)
print("✅ Saved as voice_sample.wav")