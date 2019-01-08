#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan  8 11:58:32 2019

@author: krishnaparekh
"""

import argparse
import numpy as np
import tensorflow as tf
from tensorflow.python.lib.io import file_io  # To save file of model on GCS
from io import BytesIO
import os
import numpy as np
from keras import backend as K
import keras

import model  # Your model.py file.
#import utils 


FILE_PATH = 'checkpoint.{epoch:02d}.hdf5'
CLASSIFICATION_MODEL = 'cl_model.hdf5'

def get_args():
    parser = argparse.ArgumentParser()
    # Input Arguments
    parser.add_argument(
        '--package-path',
        help = 'GCS or local path to training data',
    )
    parser.add_argument(
        '--job-dir',
        type = str,
        help = 'GCS location to write checkpoints and export models'
    )
    parser.add_argument(
        '--train-dir',
        help = 'GCS or local path to training data',
        required = True
    )
    parser.add_argument(
        '--test_data_dir',
        help = 'GCS or local path to test data',
        default = None
        )
    # Training arguments
    parser.add_argument(
        '--batch_size',
        help = 'Batch size',
        type = int,
        default = 128, 
        required = False
    )
    parser.add_argument(
        '--num_epochs',
        help = 'Number of Epochs',
        type = int,
        default = 10,
        required = False
    )
    parser.add_argument(
        '--hidden_units',
        help = 'Hidden layer sizes',
        nargs = '+',
        type = int,
        default = [32, 64, 128],
        required = False
    )
    parser.add_argument(
        '--verbosity',
        choices = ['DEBUG', 'ERROR', 'FATAL', 'INFO', 'WARN'],
        default = 'INFO'
    )
    parser.add_argument(
        '--image_size',
        default=(128,128),
        type=tuple,
        help = 'Enter tuple for image size to resize'
        )
    parser.add_argument(
      '--checkpoint-epochs',
      type=int,
      default=5,
      help='Checkpoint per n training epochs')
    
    args, _ = parser.parse_known_args()
    return args


def copy_file_to_gcs(job_dir, file_path):
    with file_io.FileIO(file_path, mode='r') as input_f:
        with file_io.FileIO(os.path.join(job_dir, file_path), mode='w+') as output_f:
            output_f.write(input_f.read())


def train_and_evaluate(args):
    '''Trains and evaluates model.
    '''
   
    f_train_i = BytesIO(file_io.read_file_to_string(args.train_dir+'/train_images.npy', binary_mode=True))
    train_images = np.load(f_train_i)   # Loading training images
    f_train_l = BytesIO(file_io.read_file_to_string(args.train_dir+'/train_labels.npy', binary_mode=True))
    train_labels = np.load(f_train_l)   # Loading training labels
    f_test_i = BytesIO(file_io.read_file_to_string(args.train_dir+'/test_images.npy', binary_mode=True))
    valid_images = np.load(f_test_i)    # Loading Validation images
    f_test_l = BytesIO(file_io.read_file_to_string(args.train_dir+'/test_labels.npy', binary_mode=True))
    valid_labels = np.load(f_test_l) 
          
    Model = model.model_cnn()
    
    ## Printing the model summary
    Model.summary()
     
    datagen = tf.keras.preprocessing.image.ImageDataGenerator(
        rotation_range=20,
        rescale=1./255,
        zoom_range=0.2,
        horizontal_flip=False)    # Generator for training dataset
    
    datagen_v = tf.keras.preprocessing.image.ImageDataGenerator()    #Generator for validation dataset
    
    Model.fit_generator(datagen.flow(train_images, train_labels, batch_size=32), \
                        steps_per_epoch=len(train_images) / 32, epochs=3, \
                        validation_data=datagen_v.flow(valid_images, valid_labels,\
                        batch_size=32), validation_steps= 3)
                                                       
    Model.save('cl_model.hdf5')
    
    job_dir = args.job_dir+'/export'
    
    if job_dir.startswith("gs://"):
        Model.save(CLASSIFICATION_MODEL)
        copy_file_to_gcs(job_dir, CLASSIFICATION_MODEL)
    else:
        Model.save(os.path.join(job_dir, CLASSIFICATION_MODEL))

##  Running the app

if __name__ == "__main__":

    args = get_args()
    arguments = args.__dict__
    train_and_evaluate(args) 
    
        