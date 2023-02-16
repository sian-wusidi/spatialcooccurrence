#!/usr/bin/env python3

import numpy as np

from tensorflow.keras import backend as K
import tensorflow as tf

  
def DiceLoss(y_true, y_pred):
    intersection = K.sum(y_true * y_pred, axis = (0,1,2)) # (batchsize, 4)
    smooth=1e-6   
    dice = (2.*intersection + smooth) / (K.sum(y_true, axis = (0,1,2)) + K.sum(y_pred, axis = (0,1,2)) + smooth)  # add also smooth to numerator, better differentiation!! because if targets = inputs = 0, the mask will be 0 and this dice won't count
    dice_mean = K.mean(dice, axis = -1)
    dice = 1 - dice_mean

    return dice
    
def Buffer_DiceLoss(y_pred, y_true, y_true_buffer):
    # mask*x_prediction1, mask*GT, mask*GT_buffer
    intersection = K.sum(y_true_buffer * y_pred, axis = (0,1,2)) # (batchsize, 4)
    smooth=1e-6   
    dice = (2.*intersection + smooth)*K.sum(y_true, axis = (0,1,2)) / (K.sum(y_true, axis = (0,1,2))**2 + K.sum(y_pred, axis = (0,1,2))**2 + smooth)  
    dice_mean = K.mean(dice, axis = -1)
    dice = 1 - dice_mean
 
    return dice
    
def priorloss(y_pred):
    # set your own prior loss here, for example:
    loss = K.max([K.sum(y_pred[:,:,:,1])/K.sum(y_pred)-0.2,K.sum(y_pred)-K.sum(y_pred)])  #wetland     

    return loss
    