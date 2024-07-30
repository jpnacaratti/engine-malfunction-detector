#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jul 20 18:10:40 2024

@author: j.nacaratti
"""

import pandas as pd
import os
from utils import download_youtube_audio, resample_audio

noise_sounds = pd.read_csv('dataset/noise_sounds.csv')

noise_dest_folder = 'dataset/noise_files'

for element in noise_sounds['youtube_id']:
    download_youtube_audio(element, noise_dest_folder)
    
    file_path = os.path.join(noise_dest_folder, f'{element}.wav')
    
    resample_audio(file_path, file_path, 16000)

    print(f'Saved audio to: {file_path}')

healthy_sounds = pd.read_csv('dataset/healthy_sounds.csv')

healthy_dest_folder = 'dataset/healthy_files'

for element in healthy_sounds['youtube_id']:
    download_youtube_audio(element, healthy_dest_folder)
    
    file_path = os.path.join(healthy_dest_folder, f'{element}.wav')
    
    resample_audio(file_path, file_path, 16000)

    print(f'Saved audio to: {file_path}')
    
print(f'Finished downloading audios: {len(os.listdir(noise_dest_folder))} | {len(os.listdir(healthy_dest_folder))}')










