#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 21:36:58 2024

@author: j.nacaratti
"""

import pandas as pd
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.callbacks import EarlyStopping
import tensorflow as tf
from utils import plot_training_history


batch_size = 80
epochs = 115
validation_size = 0.2


data = pd.read_csv('dataset/audio_features.csv')

X = data.iloc[:, 2:]
y = data.iloc[:, 1]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=validation_size, stratify=y)

model = Sequential([
    Dense(units=64, activation='relu', input_dim=128, kernel_regularizer='l2'),
    Dropout(0.3),
    Dense(units=32, activation='relu', kernel_regularizer='l2'),
    Dropout(0.3),
    Dense(units=16, activation='relu', kernel_regularizer='l2'),
    Dropout(0.3),
    Dense(units=8, activation='relu', kernel_regularizer='l2'),
    Dropout(0.3),
    Dense(units=1, activation='sigmoid')
])

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['binary_accuracy'])

early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

history = model.fit(X_train, y_train, 
                    batch_size = batch_size, 
                    epochs = epochs,
                    validation_data = (X_test, y_test),
                    callbacks = [early_stopping])

bias, accuracy = model.evaluate(X_test, y_test)

plot_training_history(history)

print(f"Accuracy: {round(accuracy * 100, 2)}")

tf.saved_model.save(model, f'models/classifier__bs_{batch_size}__epoch_{epochs}__val_{accuracy:.4f}')