#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Jul 21 18:06:41 2024

@author: j.nacaratti
"""

import os
from utils import cut_audio, AsyncAudioPlayer
import soundfile as sf
import pandas as pd

audio_length = 5 # seconds
out_samplerate = 16000
start_index = 20

noise_path = 'dataset/noise_files'

noise_data = pd.read_csv('dataset/noise_sounds.csv')

print("Started for NOISE FILES")

noise_files = os.listdir(noise_path)
noise_files_amount = len(noise_files)
for i in range(start_index, noise_files_amount):
    element = noise_files[i]
    yt_id = element.replace(".wav", "")
    
    print()
    print(f"Analising audio '{i}/{noise_files_amount}': {element}")
    
    filtered_df = noise_data[noise_data['youtube_id'] == yt_id]
    title = filtered_df['title'].iloc[0]
    
    print(f"Video title: {title}")
    
    audio_path = os.path.join(noise_path, element)
    
    count = 0

    chunks = cut_audio(audio_path, audio_length)
    for i in range(len(chunks)):
        chunk = chunks[i]
        
        while True:
            
            player = AsyncAudioPlayer()
            player.play(chunk, out_samplerate)
            
            res = input(f"Analising CHUNK = {i}. Proceed with the audio? y|w|n|r ")
            
            player.stop()
            
            if res == "y":
                # CHUNK_{YT_VIDEO_ID}_{VIDEO_ID_FILES_SAVED}_{CHUNK_SAVED}
                saved_filename = f"CHUNK__{yt_id}__{count}__{i}.wav" 
            elif res == "w":
                saved_filename = f"CHUNK__W__{yt_id}__{count}__{i}.wav"
            elif res == "n" or res == "":
                print("Playing next CHUNK...")
                break
            elif res == "r":
                print("Repeating CHUNK...")
                continue
            else:
                continue
            
            out_path = os.path.join("dataset/noise_cutted", saved_filename)
            sf.write(out_path, chunk, out_samplerate)
            count += 1
            
            print(f"CHUNK saved as: {saved_filename}")
            break
            
            
healthy_path = 'dataset/healthy_files'

healthy_data = pd.read_csv('dataset/healthy_sounds.csv')

print("Started for HEALTHY FILES")

healthy_files = os.listdir(healthy_path)
healthy_files_amount = len(healthy_files)
for i in range(start_index, healthy_files_amount):
    element = healthy_files[i]
    yt_id = element.replace(".wav", "")
    
    print()
    print(f"Analising audio '{i}/{healthy_files_amount}': {element}")
    
    filtered_df = healthy_data[healthy_data['youtube_id'] == yt_id]
    title = filtered_df['title'].iloc[0]
    duration_s = filtered_df['duration'].iloc[0]
    
    print(f"Video title: {title}")
    print(f"Duration: {duration_s}")
    
    audio_path = os.path.join(healthy_path, element)
    
    count = 0

    chunks = cut_audio(audio_path, audio_length)
    for i in range(len(chunks)):
        chunk = chunks[i]
        
        while True:
            
            player = AsyncAudioPlayer()
            player.play(chunk, out_samplerate)
            
            res = input(f"Analising CHUNK = {i}. Proceed with the audio? y|w|n|r ")
            
            player.stop()
            
            if res == "y":
                # CHUNK_{YT_VIDEO_ID}_{VIDEO_ID_FILES_SAVED}_{CHUNK_SAVED}
                saved_filename = f"CHUNK__{yt_id}__{count}__{i}.wav" 
            elif res == "w":
                saved_filename = f"CHUNK__W__{yt_id}__{count}__{i}.wav"
            elif res == "n" or res == "":
                print("Playing next CHUNK...")
                break
            elif res == "r":
                print("Repeating CHUNK...")
                continue
            else:
                continue
            
            out_path = os.path.join("dataset/healthy_cutted", saved_filename)
            sf.write(out_path, chunk, out_samplerate)
            count += 1
            
            print(f"CHUNK saved as: {saved_filename}")
            break

