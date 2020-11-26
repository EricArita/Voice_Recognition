import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import models

def voice_recognition_func(trained_model):
    AUTOTUNE = tf.data.experimental.AUTOTUNE
    commands = ['down', 'go', 'left', 'no', 'off', 'on', 'right', 'stop', 'up', 'user_voice', 'yes']

    def decode_audio(audio_binary):
        audio, _ = tf.audio.decode_wav(audio_binary)
        return tf.squeeze(audio, axis=-1)

    def get_label(file_path):
        parts = tf.strings.split(file_path, os.path.sep)
        return parts[-2]
    
    def get_spectrogram(waveform):
        zero_padding = tf.zeros([16000] - tf.shape(waveform), dtype=tf.float32)
        waveform = tf.cast(waveform, tf.float32)
        equal_length = tf.concat([waveform, zero_padding], 0)
        spectrogram = tf.signal.stft(
            equal_length, frame_length=255, frame_step=128)
        spectrogram = tf.abs(spectrogram)

        return spectrogram

    def get_waveform_and_label(file_path):
        label = get_label(file_path)
        audio_binary = tf.io.read_file(file_path)
        waveform = decode_audio(audio_binary)
        return waveform, label

    def get_spectrogram_and_label_id(audio, label):
        spectrogram = get_spectrogram(audio)
        spectrogram = tf.expand_dims(spectrogram, -1)
        label_id = tf.argmax(label == commands)
        return spectrogram, label_id

    def preprocess_dataset(files):
        files_ds = tf.data.Dataset.from_tensor_slices(files)
        output_ds = files_ds.map(
            get_waveform_and_label, num_parallel_calls=AUTOTUNE)
        output_ds = output_ds.map(
            get_spectrogram_and_label_id,  num_parallel_calls=AUTOTUNE)
        return output_ds

    path = pathlib.Path('data')
    sample_file = path/'user_voice/microphone_results.wav'
    sample_ds = preprocess_dataset([str(sample_file)])

    for spectrogram, _ in sample_ds.batch(1):
        prediction = trained_model(spectrogram)
        final_res = tf.nn.softmax(prediction[0])

    return str(commands[np.argmax(final_res)])
