#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 21:36:58 2024

@author: j.nacaratti
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential

data = pd.read_csv('dataset/audio_features.csv')

X = data.iloc[:, 2:]
y = data.iloc[:, 1]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.2, stratify=y)





