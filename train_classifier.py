#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 30 21:36:58 2024

@author: j.nacaratti
"""

import pandas as pd
from keras.callbacks import EarlyStopping
from sklearn.model_selection import train_test_split
from utils import plot_training_history
from models import get_baseline_dense_encoder, build_classifier_net


##############
# Parameters #
##############

neurons = 64
kernel_initializer = 'glorot_uniform'
kernel_regularizer = None
dropout = 0.2
batch_size = 64
epochs = 100
validation_size = 0.2
loss = 'binary_crossentropy'
optimizer = 'adam'
metrics = ['binary_accuracy']


#####################
# Creating datasets #
#####################

data = pd.read_csv('dataset/audio_features.csv')

X = data.iloc[:, 2:]
y = data.iloc[:, 1]

X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=validation_size, stratify=y)


#################
# Training loop #
#################

encoder = get_baseline_dense_encoder(neurons, kernel_initializer, kernel_regularizer, dropout)
model = build_classifier_net(encoder, loss, optimizer, metrics)

history = model.fit(
    x = X_train, 
    y = y_train, 
    batch_size = batch_size, 
    epochs = epochs,
    validation_data = (X_test, y_test),
    callbacks = [
        EarlyStopping(
            monitor='val_loss', 
            patience=10,
            min_delta = 0.001,
            restore_best_weights=True
        )
    ]
)

bias, accuracy = model.evaluate(X_test, y_test)

plot_training_history(history)

print(f"Accuracy: {round(accuracy * 100, 2)}")

model.save(f'models/classifier__bs_{batch_size}__epoch_{epochs}__val_{accuracy:.4f}', save_format = 'tf')