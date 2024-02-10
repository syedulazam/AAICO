import librosa
import numpy as np
import time
import threading
import queue
import pickle

############################ sklearn model ############################

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Define a simple classifier (you should train it on labeled data in a real scenario)
classifier = RandomForestClassifier()

############################

############################ transformers  model ############################

import torch
from transformers import Wav2Vec2ForSequenceClassification, Wav2Vec2Tokenizer

# Load pre-trained transformer model and tokenizer
model_name = "facebook/wav2vec2-base-960h"
tokenizer = Wav2Vec2Tokenizer.from_pretrained(model_name)
model = Wav2Vec2ForSequenceClassification.from_pretrained(model_name)

############################

############################ Keras sequential model #########################

from keras.models import Sequential
from keras.layers import Dense, Flatten, Dropout, BatchNormalization

############################

########### PARAMETERS ###########
# DO NOT MODIFY
# Desired sample rate 16000 Hz
sample_rate = 16000
# Frame length
frame_length = 512

########### AUDIO FILE ###########
# DO NOT MODIFY
# Path to the audio file
audio_file = "audio_aaico_challenge.wav"

# Read the audio file and resample it to the desired sample rate
audio_data, current_sample_rate = librosa.load(
    audio_file, 
    sr=sample_rate,
)
audio_data_int16 = (audio_data * 32767).astype(np.int16)
number_of_frames = len(audio_data_int16) // frame_length
audio_data_int16 = audio_data_int16[:number_of_frames * frame_length]
audio_duration = len(audio_data_int16) / sample_rate

########### STREAMING SIMULATION ###########
# DO NOT MODIFY
results = np.zeros(shape=(3, len(audio_data_int16)), dtype=np.int64)
# Detection mask lines are SENT TIME, LABEL, RECEIVE TIME.
buffer = queue.Queue()
start_event = threading.Event()


def label_samples(list_samples_id, labels):
    receive_time = time.time_ns()
    results[1][list_samples_id] = labels
    results[2][list_samples_id] = receive_time

def notice_send_samples(list_samples_id):
    send_time = time.time_ns()
    results[0][list_samples_id] = send_time

def build_model(input_shape):
    model = Sequential([
        Flatten(input_shape=input_shape),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(1, activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
    return model

def emit_data():
    time.sleep(.5)
    print('Start emitting')
    start_event.set()
    for i in range(0, number_of_frames):
        list_samples_id = np.arange(i*frame_length, (i+1)*frame_length)
        time.sleep(frame_length / sample_rate) # Simulate real time
        frame = audio_data_int16[list_samples_id]
        buffer.put(frame)
        notice_send_samples(list_samples_id)
    print('Stop emitting')

# Load the labeled dataset
with open('label_samples.pkl', 'rb') as file:
    labeled_samples = pickle.load(file)

def process_data():
    i = 0
    start_event.wait()
    print('Start processing')

    # Feature shape for the neural network
    input_shape = (number_of_frames, frame_length)

    # Build and compile the model
    model = build_model(input_shape)

    while i != number_of_frames:
        frame = buffer.get()

        # Extract features or preprocess the frame as needed
        features = frame.reshape((1, frame_length))

        # Convert features to predictions using the trained model
        inputs = tokenizer(features.tolist(), return_tensors="pt", padding="max_length", truncation=True, max_length=frame_length, stride=frame_length, sampling_rate=sample_rate)
        outputs = model(**inputs)
        predictions = outputs.logits

        # Assuming binary classification
        labels = (predictions > 0).astype(np.int)

        list_samples_id = np.arange(i*frame_length, (i+1)*frame_length)
        label_samples(list_samples_id, labels)
        i += 1

    print('Stop processing')

    # Save the list to a file
    with open('results.pkl', 'wb') as file:
        pickle.dump(results, file)

    # Save the list to a file
    with open('results.pkl', 'wb') as file:
        pickle.dump(results, file)

if __name__ == "__main__":
    time_measurement = []

    thread_process = threading.Thread(target=process_data)
    thread_emit = threading.Thread(target=emit_data)

    thread_process.start()
    thread_emit.start()
