#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug  2 23:22:55 2024

@author: j.nacaratti
"""

import os
import numpy as np
from sklearn.metrics import accuracy_score
from utils import download_youtube_audio, resample_audio, cut_audio, process_chunks, load_tflite_model, extract_embeddings_tflite, classify_tflite


#################
# Configuration #
#################

classifier_path = 'models/fault_classifier_89.tflite'
vggish_path = 'vggish/models/vggish.tflite'
validation_dir = 'dataset/validation_audios'
threshold = 0.5
out_samplerate = 16000
cut_audio_seconds = 5


###########################
# Downloading a new audio #
###########################

video_id = 'IZSApOuP_K8'
has_fault = True
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

vggish_interpreter, vggish_input_details, vggish_output_details = load_tflite_model(vggish_path)
classi_interpreter, classi_input_details, classi_output_details = load_tflite_model(classifier_path)

results = []
expected = []
for element in os.listdir(validation_dir):
    audio_path = os.path.join(validation_dir, element)
    
    for embd in extract_embeddings_tflite(vggish_interpreter, vggish_input_details, vggish_output_details, audio_path):
        
        expected.append(int(element.startswith('1')))
        
        predict = classify_tflite(classi_interpreter, classi_input_details, classi_output_details, np.expand_dims(embd, axis = 0))
        results.append(predict[0][0])

results = np.array(results)

binary_results = (results > threshold).astype(int)

val_accuracy = accuracy_score(expected, binary_results) * 100

print(f"Validation accuracy: {val_accuracy:.2f}")
 
    