import random
import os
import wave
import pyaudio
import pathlib
import numpy as np
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import layers
from tensorflow.keras import models
from IPython import display

from kivy.app import App
from kivy.uix.widget import Widget
from kivy.properties import StringProperty

import train_models

def get_voice_input():
    sample_file = 'data/mini_speech_commands/yes/0ab3b47d_nohash_0.wav'
    wf = wave.open(sample_file, 'r')

    CHUNK = 1024
    FORMAT = pyaudio.paInt16
    CHANNELS = wf.getnchannels()
    RATE = wf.getframerate()
    RECORD_SECONDS = 1
    WAVE_OUTPUT_FILENAME = "data/user_voice/microphone_results.wav"

    p = pyaudio.PyAudio()

    stream = p.open(format=FORMAT,
                     channels=CHANNELS,
                     rate=RATE,
                     input=True,
                     frames_per_buffer=CHUNK)

    print("Start recording...")

    frames = []
    for i in range(0, int(RATE / CHUNK * RECORD_SECONDS)):
        data = stream.read(CHUNK)
        frames.append(data)

    print("Done record !")

    stream.stop_stream()
    stream.close()
    p.terminate()

    wf = wave.open(WAVE_OUTPUT_FILENAME, 'wb')
    wf.setnchannels(CHANNELS)
    wf.setsampwidth(p.get_sample_size(FORMAT))
    wf.setframerate(RATE)
    wf.writeframes(b''.join(frames))
    wf.close()

class YourWidget(Widget):
    status = StringProperty()
    analyze_voice_func = lambda a: a

    def __init__(self, **kwargs):
        super(YourWidget, self).__init__(**kwargs)
        self.status = "WELLCOME"

    def train_models(self):
        self.status = "Models is being trained..."  
        self.analyze_voice_func = train_models.train()
        self.status = "Train finish, waiting for start recording..." 
    
    def record(self):
        get_voice_input()
        self.status = self.analyze_voice_func()

class Main(App):
    def build(self):
        return YourWidget()


if __name__ == '__main__':
    Main().run()
