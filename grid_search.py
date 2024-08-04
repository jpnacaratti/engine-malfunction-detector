#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 01:30:43 2024

@author: j.nacaratti
"""

import os
import logging
import pandas as pd
from itertools import product
from sklearn.model_selection import train_test_split
from keras.callbacks import EarlyStopping
from models import get_baseline_dense_encoder, build_classifier_net
from utils import plot_training_history, dict_to_json_file, convert_training_info_to_dict


logging.getLogger('tensorflow').setLevel(logging.ERROR)


##############
# Parameters #
##############

neurons = [32, 64, 96]
kernel_initializer = ['glorot_uniform']
kernel_regularizer = ['l2', None]
dropout = [0.2, 0.3]
third_block = [True]
fourth_block = [True, False]
epochs = [30, 64, 80, 100, 115, 130]
batch_size = [32, 48, 64, 80, 96]
validation_size = 0.2
loss = 'binary_crossentropy'
optimizer = 'adam'
metrics = ['binary_accuracy']
num_workers = 6
use_multiprocessing = True
save_folder = 'grid_search'

n_comb = len(neurons) * len(kernel_initializer) * len(kernel_regularizer) * len(dropout) * len(fourth_block) * len(third_block) * len(epochs) * len(batch_size)
print(f"Number of possible combinations: {n_comb}")


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

os.makedirs(save_folder, exist_ok=True)

best_combo = None
best_val_acc = 0.0
best_val_loss = 0.0

historic = {}

combos = product(neurons, kernel_initializer, kernel_regularizer, dropout, third_block, fourth_block, epochs, batch_size)

count = 0
for combo in combos:
    neurons, kernel_init, kernel_reg, dropout, third_block, fourth_block, epochs, batch_size = combo
    print()
    print(f"Training with ({count}): neurons={neurons}, kernel_init={kernel_init}, kernel_reg={kernel_reg}, dropout={dropout}, third_block={third_block}, fourth_block={fourth_block}, epochs={epochs}, batch_size={batch_size}")

    encoder = get_baseline_dense_encoder(neurons, kernel_init, kernel_reg, dropout, third_block=third_block, fourth_block=fourth_block)
    model = build_classifier_net(encoder, loss, optimizer, metrics)
    
    fit_hist = model.fit(
        x = X_train,
        y = y_train,
        batch_size = batch_size,
        epochs = epochs,
        validation_data = (X_test, y_test),
        workers = num_workers,
        use_multiprocessing = use_multiprocessing,
        verbose = 0,
        callbacks = [
            EarlyStopping(
                monitor='val_loss', 
                patience=10,
                min_delta = 0.001,
                restore_best_weights=True
            )
        ]
    )
    
    accuracy = fit_hist.history['binary_accuracy'][-1] * 100.0
    val_accuracy = model.evaluate(X_test, y_test, verbose = 0)[1] * 100.0
    val_loss = min(fit_hist.history['val_loss'])
    
    print(f"Finished training, accuracy: {accuracy:.4f}")
    
    model_name = f'val_{val_accuracy:.4f}__loss_{val_loss:.4f}__index_{count}'
    save_path = os.path.join(save_folder, model_name)
    
    result = convert_training_info_to_dict(accuracy, val_accuracy, val_loss, save_path, combo)
    
    historic[f'{count}'] = result
    
    model.save(save_path, save_format = 'tf')
    plot_training_history(history = fit_hist, save_path = f'{save_path}/assets/training_history.png')
    dict_to_json_file(result, save_path = f'{save_path}/assets/params.json')
    
    count += 1
    
    if best_val_acc < val_accuracy and (best_val_loss > val_loss or best_val_loss == 0.0):
        print(f"Val accuracy increased from: {best_val_acc:.4f} to {val_accuracy:.4f}")
        print(f"Loss increased from: {best_val_loss} to {val_loss}")
        
        best_val_acc = val_accuracy
        best_val_loss = val_loss
        
        best_combo = combo
    else:
        print(f"Val accuracy did not increase from: {best_val_acc:.4f}")
        print(f"Loss did not increase from: {best_val_loss}")

print("Finished Grid Search!")

dict_to_json_file(historic, save_path = 'grid_search.json')

print(f"Best params: {best_combo}")