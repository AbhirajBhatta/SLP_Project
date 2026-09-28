"""
generate_test_audio.py
Utility script to generate sample multi-speaker audio WAV file for testing.
"""

import wave
import numpy as np

def generate_test_wav(filename="sample_meeting.wav", duration=12.0, sample_rate=16000):
    t = np.linspace(0, duration, int(sample_rate * duration), False)
    
    # Speaker 1 (200Hz tone sequence for 4s)
    s1 = np.where((t >= 0.5) & (t <= 4.0), 0.3 * np.sin(2 * np.pi * 200 * t), 0)
    
    # Speaker 2 (350Hz tone sequence for 4s)
    s2 = np.where((t >= 4.5) & (t <= 8.5), 0.3 * np.sin(2 * np.pi * 350 * t), 0)
    
    # Speaker 1 (200Hz tone sequence for 3s)
    s3 = np.where((t >= 9.0) & (t <= 11.5), 0.3 * np.sin(2 * np.pi * 200 * t), 0)

    audio = s1 + s2 + s3
    pcm_bytes = (audio * 32767).astype(np.int16).tobytes()

    with wave.open(filename, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        wf.writeframes(pcm_bytes)

    print(f"Generated sample WAV file: {filename} ({duration} seconds)")

if __name__ == "__main__":
    generate_test_wav()
