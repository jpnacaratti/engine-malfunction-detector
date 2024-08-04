#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 18:06:41 2024

@author: j.nacaratti
"""

import os
import pandas as pd
from utils import cut_audio, process_chunks


#################
# Configuration #
#################

audio_length = 5 # seconds
out_samplerate = 16000
start_index = 0


#########################
# Selecting noise files #
#########################

noise_path = 'dataset/noise_files'
noise_data = pd.read_csv('dataset/noise_sounds_ids.csv')

print("Started for NOISE FILES")

noise_files = os.listdir(noise_path)
noise_files_amount = len(noise_files)
for i in range(start_index, noise_files_amount):
    element = noise_files[i]
    yt_id = element.replace(".wav", "")
    
    print()
    print(f"Analising audio '{i}/{noise_files_amount - 1}': {element}")
    
    filtered_df = noise_data[noise_data['youtube_id'] == yt_id]
    title = filtered_df['title'].iloc[0]
    
    print(f"Video title: {title}")
    
    audio_path = os.path.join(noise_path, element)

    chunks = cut_audio(audio_path, audio_length)
    
    process_chunks(chunks, out_samplerate, yt_id, 'dataset/noise_cutted')
            

###########################
# Selecting healthy files #
###########################

healthy_path = 'dataset/healthy_files'
healthy_data = pd.read_csv('dataset/healthy_sounds_ids.csv')

print("Started for HEALTHY FILES")

healthy_files = os.listdir(healthy_path)
healthy_files_amount = len(healthy_files)
for i in range(start_index, healthy_files_amount):
    element = healthy_files[i]
    yt_id = element.replace(".wav", "")
    
    print()
    print(f"Analising audio '{i}/{healthy_files_amount - 1}': {element}")
    
    filtered_df = healthy_data[healthy_data['youtube_id'] == yt_id]
    title = filtered_df['title'].iloc[0]
    duration_s = filtered_df['duration'].iloc[0]
    
    print(f"Video title: {title}")
    print(f"Duration: {duration_s}")
    
    audio_path = os.path.join(healthy_path, element)

    chunks = cut_audio(audio_path, audio_length)
    
    process_chunks(chunks, out_samplerate, yt_id, 'dataset/healthy_cutted')

