#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 22:37:17 2024

@author: j.nacaratti
"""

import os
import numpy as np
from sklearn.metrics import accuracy_score
from keras.models import load_model
from utils import download_youtube_audio, resample_audio, cut_audio, process_chunks, extract_embeddings


#################
# Configuration #
#################

model_path = 'models/classifier__bs_80__epoch_115__val_0.8783'
vggish_model_path = 'vggish/models/vggish_model.ckpt'
validation_dir = 'dataset/validation_audios'
threshold = 0.5
out_samplerate = 16000
cut_audio_seconds = 5


###########################
# Downloading a new audio #
###########################

video_id = '7W2Ny4wlq5I'
has_fault = False
delete_original = True
download_youtube_audio(video_id, '.')


#######################
# Preprocessing audio #
#######################

resample_audio(f'{video_id}.wav', f'{video_id}.wav', out_samplerate)

chunks = cut_audio(f'{video_id}.wav', cut_audio_seconds)

process_chunks(chunks, out_samplerate, video_id, validation_dir, prefix = f'{int(has_fault)}_CHUNK__')

if delete_original:
    os.remove(f'{video_id}.wav')


###############
# Classifying #
###############

embeddings = []
expected = []
for element in os.listdir(validation_dir):
    audio_path = os.path.join(validation_dir, element)
    
    for embd in extract_embeddings(audio_path, 0.96, vggish_model_path):
        embeddings.append(embd.tolist())
        expected.append(int(element.startswith('1')))
    
embeddings = np.array(embeddings)
    
model = load_model(model_path)

results = model.predict(embeddings)

binary_results = (results > threshold).astype(int)

val_accuracy = accuracy_score(expected, binary_results) * 100

print(f"Validation accuracy: {val_accuracy:.2f}")
