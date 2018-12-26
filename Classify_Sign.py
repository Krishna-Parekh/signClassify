#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec 25 10:58:11 2018

@author: krishnaparekh
"""

import numpy as np
import os
from keras.preprocessing import image
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

from sklearn.preprocessing import LabelEncoder, OneHotEncoder
from keras.utils import np_utils  # For One Hot Encoding

def generate_test(image_dir):
    
    directory = image_dir   
    dataset_full = []    
    folders = next(os.walk(directory))
    count_fol = 0
    count_file = 0
    for fol in folders[1]:
        fol_path = directory + fol + "/"
        count_fol += 1
        files = next(os.walk(fol_path))[2]        
        for f in files:
            file_lbl = []
            file_path = fol_path + f
            file_lbl.append(file_path)
            file_lbl.append(fol)
            count_file += 1
            dataset_full.append(file_lbl)

    image_rows = 100  # Size of the image
    image_cols = 100    
    le = LabelEncoder()
           
    test_images = []
    test_labels = ["" for _ in range(0, len(dataset_full))]
           
    for i in range(0, len(dataset_full)):
        img = image.load_img(dataset_full[i][0])
        img = img.resize((image_rows, image_cols), Image.ANTIALIAS)   # Resize fist two dimensions
        img_ar = np.array(img)
        lbl = dataset_full[i][1]
        test_images.append(img_ar)
        test_labels[i] = lbl

    test_images = np.array(test_images)
    test_labels = le.fit_transform(test_labels)
    test_labels = np_utils.to_categorical(test_labels)

    return test_images, test_labels


from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.layers.normalization import BatchNormalization

# Layer Values
num_filters = 32            # No. of conv filters
max_pool_size = (2,2)       # shape of max_pool
conv_kernel_size = (3, 3)    # conv kernel shape
imag_shape = (100, 100, 3)
num_classes = 2
drop_prob = 0.5             # fraction to drop (0-1.0)

# Define model type
model = Sequential()

# 1st Layer
model.add(Conv2D(num_filters, conv_kernel_size[0], conv_kernel_size[1], input_shape=imag_shape, activation='relu'))
model.add(MaxPooling2D(pool_size=max_pool_size))

# 2nd Convolution Layer
model.add(Conv2D(num_filters*2, conv_kernel_size[0], conv_kernel_size[1], input_shape=imag_shape, activation='relu'))
model.add(MaxPooling2D(pool_size=max_pool_size))

# 3nd Convolution Layer
model.add(Conv2D(num_filters*4, conv_kernel_size[0], conv_kernel_size[1], input_shape=imag_shape, activation='relu'))
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

# Training settings
batch_size = 128
num_epoch = 5

train_images = np.load('/home/krishnaparekh/icicprulife/Signature/train_images.npy')   # Loading training images
train_labels = np.load('/home/krishnaparekh/icicprulife/Signature/train_labels.npy')   # Loading training labels
test_images = np.load('/home/krishnaparekh/icicprulife/Signature/test_images.npy')    # Loading Validation images
test_labels = np.load('/home/krishnaparekh/icicprulife/Signature/test_labels.npy')   # Loading validation labels

datagen = image.ImageDataGenerator(
    rotation_range=20,
    rescale=1./255,
    zoom_range=0.2,
    horizontal_flip=False)    # Generator for training dataset

datagen_v = image.ImageDataGenerator()    #Generator for validation dataset

model.fit_generator(datagen.flow(train_images, train_labels, batch_size=32), \
                    steps_per_epoch=len(train_images) / 32, epochs=num_epoch, \
                    validation_data=datagen_v.flow(test_images, test_labels, batch_size=32), validation_steps= 3)

test_images_t, test_labels_t = generate_test("/home/krishnaparekh/icicprulife/Signature/Data_test/")

test_pred_t = model.predict(test_images_t)

test_label_t = np.argmax(test_labels_t, axis=1)
test_pred_t = np.argmax(test_pred_t,axis=1)

from sklearn.metrics import precision_score, recall_score, accuracy_score

print("Precision ", precision_score(test_label_t,test_pred_t))
print("Recall ",  recall_score(test_label_t,test_pred_t))
print("Accuracy ", accuracy_score(test_label_t,test_pred_t))


