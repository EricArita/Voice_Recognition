import kivy
kivy.require('1.9.1')

import os
import wave
import pyaudio  

from tensorflow.keras import models

from kivy.app import App
from kivy.uix.widget import Widget
from kivy.properties import StringProperty

from voice_recognition_module import voice_recognition_func

def get_voice_input():
    sample_file = 'data/sample/00f0204f_nohash_0.wav'
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
    trained_model = None

    def __init__(self, **kwargs):
        super(YourWidget, self).__init__(**kwargs)
        self.trained_model = models.load_model('saved_model/audio_model')
        self.status = "WELLCOME"
    
    def record(self):
        get_voice_input()
        self.status = voice_recognition_func(self.trained_model)

class MainApp(App):
    def build(self):
        return YourWidget()


if __name__ == '__main__':
    MainApp().run()
