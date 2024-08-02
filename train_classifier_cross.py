#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug  1 21:46:01 2024

@author: j.nacaratti
"""

import pandas as pd
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.wrappers.scikit_learn import KerasClassifier
from sklearn.model_selection import cross_val_score


batch_size = 16
epochs = 30


data = pd.read_csv('dataset/audio_features.csv')

X = data.iloc[:, 2:]
y = data.iloc[:, 1]

def create_network():
    model = Sequential([
        Dense(units=64, activation='relu', input_dim=128, kernel_regularizer='l2'),
        Dropout(0.3),
        Dense(units=32, activation='relu', kernel_regularizer='l2'),
        Dropout(0.3),
        Dense(units=16, activation='relu', kernel_regularizer='l2'),
        Dropout(0.3),
        Dense(units=1, activation='sigmoid')
    ])
    
    model.compile(optimizer = 'adam', loss = 'binary_crossentropy', 
                  metrics = ['binary_accuracy'])
    
    return model

classifier = KerasClassifier(build_fn = create_network, 
                             epochs = epochs,
                             batch_size = batch_size)

results = cross_val_score(estimator = classifier, 
                          X = X, y = y, 
                          cv = 10, scoring = 'accuracy')

print(f"Final accuracy: {results.mean():.4f}")
print(f"Standard deviation: {results.std():.4f}")