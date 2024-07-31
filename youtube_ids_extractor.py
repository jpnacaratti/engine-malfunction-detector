#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 17 21:46:14 2024

@author: j.nacaratti
"""

import googleapiclient.discovery
import pandas as pd
from utils import fetch_youtube_ids

DEV_API_KEY = "YOUR_KEY_HERE"

youtube = googleapiclient.discovery.build("youtube", "v3", developerKey=DEV_API_KEY)

noise_queries = [
    'weird engine sound',
    'engine buzzing/farting sound',
    'car engine noise',
    'Engine knocking',
    'Engine bad sound',
    'Engine Whining Noise',
    'engine blown sounds',
    'engine tick sound',
    'engine Clicking Noise',
    'engine rattle noise',
    'engine Ticking Noise',
]

healthy_queries = [
  'scania engine sound',
  'diesel truck ensine sound',
  'truck engine sound'
]

#############################
# Fetch Noise Engine Sounds #
#############################

ids, titles, durations = fetch_youtube_ids(queries = noise_queries, max_results = 10, yt_service = youtube)

noise_sounds = pd.DataFrame({'youtube_id': ids, 'title': titles, 'duration': durations})

noise_sounds.to_csv('dataset/noise_sounds_ids.csv', index = False)

###############################
# Fetch Healthy Engine Sounds #
###############################

ids, titles, durations = fetch_youtube_ids(queries = healthy_queries, max_results = 10, yt_service = youtube)

healthy_sounds = pd.DataFrame({'youtube_id': ids, 'title': titles, 'duration': durations})

healthy_sounds.to_csv('dataset/healthy_sounds_ids.csv', index = False)







