#!/usr/bin/env python3

import os.path

import numpy as np
import keras
import shutil


class DataGeneratorall(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, data_location, batch_size=32, shuffle=True):
        'Initialization'
        
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.shuffle = shuffle
        self.data_location = data_location
        self.on_epoch_end()
    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        X, y, X1 = self.__data_generation(list_IDs_temp)
        # y = self.__data_generation(list_IDs_temp)
        return X, y, X1

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        Lab = []
        Anno = []
        Unlab = []
        for i, ID in enumerate(list_IDs_temp):
            sheet_ID_lab = ID.split(",")[1]
            anno_ID_lab = ID.split(",")[0]
            sheet_ID_unlab = ID.split(",")[2]

            lab = np.load(self.data_location + sheet_ID_lab)
            Lab.append(lab['arr_0'][:, :, 0:3])
            
            anno = np.load(self.data_location + anno_ID_lab)
            Anno.append(anno['arr_0'])

            unlab = np.load(self.data_location + sheet_ID_unlab)
            Unlab.append(unlab['arr_0'][:, :, 0:3])
            
        Lab = np.asarray(Lab)
        Anno = np.asarray(Anno)
        Unlab = np.asarray(Unlab)

        return Lab, Anno, Unlab

class DataGenerator(keras.utils.Sequence):
    'Generates data for Keras'
    def __init__(self, list_IDs, data_location, batch_size=32, labelled = True, shuffle=True):
        'Initialization'
        
        self.batch_size = batch_size
        self.list_IDs = list_IDs
        self.labelled = labelled
        self.shuffle = shuffle
        self.data_location = data_location
        self.on_epoch_end()

    def __len__(self):
        'Denotes the number of batches per epoch'
        return int(np.floor(len(self.list_IDs) / self.batch_size))

    def __getitem__(self, index):
        'Generate one batch of data'
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:(index+1)*self.batch_size]

        # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        # Generate data
        x, y = self.__data_generation(list_IDs_temp)
        # y = self.__data_generation(list_IDs_temp)
        return x, y

    def on_epoch_end(self):
        'Updates indexes after each epoch'
        self.indexes = np.arange(len(self.list_IDs))
        if self.shuffle == True:
            np.random.shuffle(self.indexes)

    def __data_generation(self, list_IDs_temp):
        'Generates data containing batch_size samples' # X : (n_samples, *dim, n_channels)
        if self.labelled == False:
            Unlab = []
            for i, ID in enumerate(list_IDs_temp):
                sheet_ID = ID
                unlab = np.load(self.data_location + sheet_ID)
                Unlab.append(unlab['arr_0'][:, :, 0:3])   
            Unlab = np.asarray(Unlab)
            Anno = None
            return Unlab, Anno
        else:
            Lab = []
            Anno = []
            for i, ID in enumerate(list_IDs_temp):
                sheet_ID = ID.split(",")[1]
                anno_ID = ID.split(",")[0]
                lab = np.load(self.data_location + sheet_ID)
                Lab.append(lab['arr_0'][:, :, 0:3])
                anno = np.load(self.data_location + anno_ID)
                Anno.append(anno['arr_0'])
            Lab = np.asarray(Lab)
            Anno = np.asarray(Anno)
            return Lab, Anno
        