import os
import wave
import pyaudio
import pathlib
import numpy as np

sample_file = 'data/mini_speech_commands/yes/0ab3b47d_nohash_0.wav'
wf = wave.open(sample_file, 'r')

CHUNK = 1024
FORMAT = pyaudio.paInt16
CHANNELS = wf.getnchannels()
RATE = wf.getframerate()
RECORD_SECONDS = 1
WAVE_OUTPUT_FILENAME = "data/data_test/yes/microphone_results.wav"

p = pyaudio.PyAudio()

stream = p.open(format=FORMAT,
                channels=CHANNELS,
                rate=RATE,
                input=True,
                frames_per_buffer=CHUNK)

print("* recording")

frames = []

for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
    data = stream.read(CHUNK)
    frames.append(data)

print("* done recording")

stream.stop_stream()
stream.close()
p.terminate()

wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
wf.setnchannels(CHANNELS)
wf.setsampwidth(p.get_sample_size(FORMAT))
wf.setframerate(RATE)
wf.writeframes(b''.join(frames))
wf.close()
