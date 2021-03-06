import os
import pathlib

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import tensorflow as tf

from tensorflow.keras.layers.experimental import preprocessing
from tensorflow.keras import layers
from tensorflow.keras import models
from IPython import display

def decode_audio(audio_binary):
    audio, _ = tf.audio.decode_wav(audio_binary)
    return tf.squeeze(audio, axis=-1)


def get_label(file_path):
    parts = tf.strings.split(file_path, os.path.sep)
    print(parts)
    return parts[-2]


def get_waveform_and_label(file_path):
    label = get_label(file_path)
    print(label)
    audio_binary = tf.io.read_file(file_path)
    waveform = decode_audio(audio_binary)
    return waveform, label


def train():
    data_dir = pathlib.Path('data\mini_speech_commands')
    # if not data_dir.exists():
    #   tf.keras.utils.get_file(
    #       'mini_speech_commands.zip',
    #       origin="http://storage.googleapis.com/download.tensorflow.org/data/mini_speech_commands.zip",
    #       extract=True,
    #       cache_dir='.', cache_subdir='data')

    commands = np.array(tf.io.gfile.listdir(str(data_dir)))
    commands = commands[commands != 'README.md']

    filenames = tf.io.gfile.glob(str(data_dir) + '/*/*')
    filenames = tf.random.shuffle(filenames)

    train_files = filenames[:6400]
    val_files = filenames[6400: 6400 + 800]
    test_files = filenames[-800:]

    AUTOTUNE = tf.data.experimental.AUTOTUNE
    files_ds = tf.data.Dataset.from_tensor_slices(train_files)
    waveform_ds = files_ds.map(
        get_waveform_and_label, num_parallel_calls=AUTOTUNE)

    rows = 3
    cols = 3
    n = rows*cols
    _, axes = plt.subplots(rows, cols, figsize=(10, 12))
    for i, (audio, label) in enumerate(waveform_ds.take(n)):
        r = i // cols
        c = i % cols
        ax = axes[r][c]
        ax.plot(audio.numpy())
        ax.set_yticks(np.arange(-1.2, 1.2, 0.2))
        label = label.numpy().decode('utf-8')
        ax.set_title(label)

    # plt.show()

    def get_spectrogram(waveform):
        zero_padding = tf.zeros([16000] - tf.shape(waveform), dtype=tf.float32)
        waveform = tf.cast(waveform, tf.float32)
        equal_length = tf.concat([waveform, zero_padding], 0)
        spectrogram = tf.signal.stft(
            equal_length, frame_length=255, frame_step=128)
        spectrogram = tf.abs(spectrogram)

        return spectrogram

    for waveform, label in waveform_ds.take(1):
        label = label.numpy().decode('utf-8')
        spectrogram = get_spectrogram(waveform)

    display.display(display.Audio(waveform, rate=16000))

    def plot_spectrogram(spectrogram, ax):
        log_spec = np.log(spectrogram.T)
        height = log_spec.shape[0]
        X = np.arange(16000, step=height + 1)
        Y = range(height)
        ax.pcolormesh(X, Y, log_spec)

    def get_spectrogram_and_label_id(audio, label):
        spectrogram = get_spectrogram(audio)
        spectrogram = tf.expand_dims(spectrogram, -1)
        label_id = tf.argmax(label == commands)
        return spectrogram, label_id

    spectrogram_ds = waveform_ds.map(
        get_spectrogram_and_label_id, num_parallel_calls=AUTOTUNE)

    rows = 3
    cols = 3
    n = rows*cols
    _, axes = plt.subplots(rows, cols, figsize=(10, 10))
    for i, (spectrogram, label_id) in enumerate(spectrogram_ds.take(n)):
        r = i // cols
        c = i % cols
        ax = axes[r][c]
        plot_spectrogram(np.squeeze(spectrogram.numpy()), ax)
        ax.set_title(commands[label_id.numpy()])
        ax.axis('off')

    # plt.show()

    def preprocess_dataset(files):
        files_ds = tf.data.Dataset.from_tensor_slices(files)
        output_ds = files_ds.map(
            get_waveform_and_label, num_parallel_calls=AUTOTUNE)
        output_ds = output_ds.map(
            get_spectrogram_and_label_id,  num_parallel_calls=AUTOTUNE)
        return output_ds

    train_ds = spectrogram_ds
    val_ds = preprocess_dataset(val_files)
    test_ds = preprocess_dataset(test_files)

    batch_size = 64
    train_ds = train_ds.batch(batch_size)
    val_ds = val_ds.batch(batch_size)

    train_ds = train_ds.cache().prefetch(AUTOTUNE)
    val_ds = val_ds.cache().prefetch(AUTOTUNE)

    for spectrogram, _ in spectrogram_ds.take(1):
        input_shape = spectrogram.shape
        num_labels = len(commands)

# Setup for trainning
    norm_layer = preprocessing.Normalization()
    norm_layer.adapt(spectrogram_ds.map(lambda x, _: x))

    model = models.Sequential([
        layers.Input(shape=input_shape),
        preprocessing.Resizing(32, 32),
        norm_layer,
        layers.Conv2D(32, 3, activation='relu'),
        layers.Conv2D(64, 3, activation='relu'),
        layers.MaxPooling2D(),
        layers.Dropout(0.25),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_labels),
    ])

    model.summary()

    model.compile(
        optimizer=tf.keras.optimizers.Adam(),
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=['accuracy'],
    )

####################### Start tranning models ######################
    EPOCHS = 15
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=EPOCHS,
        callbacks=tf.keras.callbacks.EarlyStopping(verbose=1, patience=2),
    )

    model.save('saved_model/audio_model') # Save the entire model as a SavedModel.

######################## Analyze results and accuracy on test_ds ###############

    # metrics = history.history  
    # plt.plot(history.epoch, metrics['loss'], metrics['val_loss'])
    # plt.legend(['loss', 'val_loss'])
    # plt.show()

    model = models.load_model('saved_model/audio_model')

    test_audio = []
    test_labels = []

    for audio, label in test_ds:
        test_audio.append(audio.numpy())
        test_labels.append(label.numpy())

    test_audio = np.array(test_audio)
    test_labels = np.array(test_labels)

    y_pred = np.argmax(model.predict(test_audio), axis=1)
    y_true = test_labels

    test_acc = sum(y_pred == y_true) / len(y_true)
    print(f'Test set accuracy: {test_acc:.0%}')

    confusion_mtx = tf.math.confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(confusion_mtx, xticklabels=commands, yticklabels=commands,
                annot=True, fmt='g')
    plt.xlabel('Prediction')
    plt.ylabel('Label')
    plt.show()

    def start_recognition():
        trained_model = models.load_model('saved_model/audio_model')

        path = pathlib.Path('data')
        sample_file =  path/'user_voice/microphone_results.wav'
        sample_ds = preprocess_dataset([str(sample_file)])

        for spectrogram, label_id in sample_ds.batch(1):
            prediction = trained_model(spectrogram)
            final_res = tf.nn.softmax(prediction[0])
            plt.bar(commands, final_res)
            plt.title(f'Predictions for "{commands[label_id[0]]}"')
            plt.show()
        
        return str(commands[np.argmax(final_res)])

    return start_recognition

train()
