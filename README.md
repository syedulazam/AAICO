Overview

In this task, you should complete the 'aaico_voice_processing_challenge.py' file. This file simulates the streaming of the 'audio_aaico_challenge.wav' audio file. Frame by frame, the "emit_data" thread emits the data of the audio file. Each frame consists of 512 samples, with a sample rate of 16000 Hz for the audio file.

The "process_data" thread receives these frames. Your task is to complete the code in this thread to label each received sample and save your label using the provided function "label_samples". A sample should be labeled 0 if it is detected as a command, otherwise 1 (we consider that everything that is not a command should be broadcast).

Once the code is executed, a 'results.pkl' file will be saved, which is an array containing for each sample:

The time at which the sample was emitted. The label you assigned to the sample. The time at which the sample was labelled.

The aim is to use an ML model to get the most accurate labellings.
