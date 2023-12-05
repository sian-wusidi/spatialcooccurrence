#!/usr/bin/env python3

import sys
import pickle

import matplotlib.pyplot as plt
import numpy as np

from keras.callbacks import TensorBoard
from keras.optimizers import RMSprop, Adam

from UNETGAN.models import create_models, build_graph_all
from UNETGAN.trainingGAN import fit_models
from UNETGAN.data import DataGeneratorall, DataGenerator
from UNETGAN.losses import DiceLoss
from datetime import datetime
from keras import backend as K
import tensorflow as tf
from numpy.random import seed
from tensorflow import set_random_seed
K.clear_session()

#for tensorflow 1*
tf.device("/gpu:0") 
config = tf.ConfigProto() # for tf 2 tf.compat.v1.ConfigProto()
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU

sess = tf.Session(config=config)  # for tf 2 tf.compat.v1.Session
init = tf.global_variables_initializer()
sess.run(init)
#seed(0)
#set_random_seed(0)

def set_trainable(model, trainable):
    model.trainable = trainable
    for layer in model.layers:
        layer.trainable = trainable


def main():
    batch_size = 1  # 16 in the paper
    epochs = 50
    classifier, intermediate, discriminator = create_models(n_channels=3, wdecay = 1e-5) 
    attention, combined = build_graph_all(classifier, intermediate, discriminator) 

    opt1 = Adam(lr=0.0004, decay = 0.0001) 
    opt2 = Adam(lr=0.0001, decay = 0.0001) 
    classifier.compile(loss = DiceLoss, optimizer = opt1, metrics = ['acc'])
    attention.compile(optimizer = opt1, metrics = ['acc']*2)
    set_trainable(discriminator, False)
    combined.compile(loss = ['binary_crossentropy'], optimizer = opt2, metrics = ['acc'])
    set_trainable(discriminator, True)
    discriminator.compile(loss = ['binary_crossentropy'], optimizer = opt2, metrics = ['acc'])
    
    classifier.summary()
    attention.summary()
    discriminator.summary()
    combined.summary()
    
    data_location = "Datasamples/" 

    with open(data_location + "/paired_labelled_unlabelled.txt") as file:
        lines = file.readlines()
        training_samples_labelled_unlabelled = [line.rstrip() for line in lines]
    
    with open(data_location + "/labelled.txt") as file:
        lines = file.readlines()
        training_samples_labelled = [line.rstrip() for line in lines]
    
    with open(data_location + "/unlabelled.txt") as file:
        lines = file.readlines()
        training_samples_unlabelled = [line.rstrip() for line in lines]
        
    num_unlabelled_images = len(training_samples_unlabelled)
    num_labelled_images = len(training_samples_labelled)
    num_paired_images = len(training_samples_labelled_unlabelled)    
    training_generator_labelled_unlabelled = DataGeneratorall(training_samples_labelled_unlabelled, data_location,  batch_size = batch_size,  shuffle = True)
    training_generator_labelled = DataGenerator(training_samples_labelled,  data_location,  batch_size = batch_size, labelled = True, shuffle = True)
    training_generator_unlabelled = DataGenerator(training_samples_unlabelled,  data_location,  batch_size = batch_size, labelled = False, shuffle = True)

    writer = tf.summary.FileWriter("logs/"+ datetime.now().strftime("%Y%m%d-%H%M%S"))
    print("start training of ", num_labelled_images, " labelled images and ", num_unlabelled_images, "unlabelled images, for epochs ",  epochs, "and paired are, ", num_paired_images)
    fit_models(classifier, attention, discriminator, combined, training_generator_labelled_unlabelled, training_generator_labelled, training_generator_unlabelled, batch_size, epochs, num_labelled_images, num_unlabelled_images, num_paired_images,  writer)
    

if __name__ == '__main__':
    main()
