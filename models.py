#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Aug  3 01:41:24 2024

@author: j.nacaratti
"""

from keras.models import Sequential
from keras.layers import Dense, Dropout

def get_baseline_dense_encoder(units, kernel_initializer, kernel_regularizer, dropout, second_block = True, third_block = True, fourth_block = True):
    
    encoder = Sequential()
    
    encoder.add(Dense(units = units, activation = 'relu', kernel_initializer = kernel_initializer, kernel_regularizer = kernel_regularizer, input_dim = 128))
    encoder.add(Dropout(dropout))
    
    if second_block:
        encoder.add(Dense(units = (units // 2), activation = 'relu', kernel_initializer = kernel_initializer, kernel_regularizer = kernel_regularizer))
        encoder.add(Dropout(dropout))
    
    if third_block:
        encoder.add(Dense(units = (units // 4), activation = 'relu', kernel_initializer = kernel_initializer, kernel_regularizer = kernel_regularizer))
        encoder.add(Dropout(dropout))
    
    if fourth_block:
        encoder.add(Dense(units = (units // 8), activation = 'relu', kernel_initializer = kernel_initializer, kernel_regularizer = kernel_regularizer))
        encoder.add(Dropout(dropout))
        
    return encoder

def build_classifier_net(encoder, loss, optimizer, metrics):
    
    encoder.add(Dense(units=1, activation='sigmoid'))
    encoder.compile(optimizer = optimizer, loss = loss, metrics = metrics)
    
    return encoder
    