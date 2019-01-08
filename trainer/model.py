#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 13:20:51 2019

@author: krishnaparekh
"""

import keras
from keras import backend as K
from keras import layers, models
from keras.backend import relu, sigmoid
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, Activation, Flatten
from tensorflow.keras.layers import Conv2D, MaxPooling2D


def model_cnn(imag_shape = (100, 100, 3)):
    # Layer Values
    num_filters = 32            # No. of conv filters
    max_pool_size = (2,2)       # shape of max_pool
    conv_kernel_size = (3, 3)    # conv kernel shape
    imag_shape = imag_shape
    num_classes = 2
    drop_prob = 0.5             # fraction to drop (0-1.0)
    
    # Define model type
    model = Sequential()
    
    # 1st Layer
    model.add(Conv2D(filters=num_filters, kernel_size=(conv_kernel_size), input_shape=imag_shape, activation='relu'))
    model.add(MaxPooling2D(pool_size=max_pool_size))
    
    # 2nd Convolution Layer
    model.add(Conv2D(filters = num_filters*2, kernel_size=(conv_kernel_size), input_shape=imag_shape, activation='relu'))
    model.add(MaxPooling2D(pool_size=max_pool_size))
    
    # 3nd Convolution Layer
    model.add(Conv2D(filters = num_filters*4, kernel_size=(conv_kernel_size), input_shape=imag_shape, activation='relu'))
    model.add(MaxPooling2D(pool_size=max_pool_size))
    
    #Fully Connected Layer
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))  #Fully connected layer 
    
    # Dropout some neurons to reduce overfitting
    model.add(Dropout(drop_prob))
    
    #Readout Layer
    model.add(Dense(num_classes, activation='sigmoid'))
    
    # Set loss and measurement, optimizer, and metric used to evaluate loss
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    return model

