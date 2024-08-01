#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 31 20:20:51 2024

@author: j.nacaratti
"""

import pandas as pd
from utils import extract_all_features_from_folder


#################
# Configuration #
#################

embedding_size = 128
hop_size = 0.96 # seconds
cutted_noises_path = 'dataset/noise_cutted'
cutted_healthy_path = 'dataset/healthy_cutted'

df_columns = ['audio_path', 'has_noise'] + [f'feature_{i}' for i in range(embedding_size)]


##############
# Extracting #
##############

print("Extracting for NOISE CUTTED FILES...")
noise_features = extract_all_features_from_folder(cutted_noises_path, True, hop_size)

print("Extracting for HEALTHY CUTTED FILES...")
healthy_features = extract_all_features_from_folder(cutted_healthy_path, False, hop_size)

df_rows = []

df_rows.extend(noise_features)
df_rows.extend(healthy_features)

df = pd.DataFrame(columns=df_columns)

if df_rows:
    new_df = pd.DataFrame(df_rows, columns=df_columns)
    df = pd.concat([df, new_df], ignore_index=True)

df.to_csv('dataset/audio_features.csv', index=False)