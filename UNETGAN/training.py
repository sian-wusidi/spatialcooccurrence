#!/usr/bin/env python3

import numpy as np
import tensorflow as tf

def fit_models(classifier, attention, training_generator_labelled_unlabelled, training_generator_labelled, batch_size, epochs, num_labelled_images, num_paired_images,  writer):
    
    steps_per_epoch_labelled = num_labelled_images // batch_size
    steps_per_epoch_paired = num_paired_images // batch_size
    seg_loss = 0
    atten_loss = 0
    for e in range(epochs):
        # train segmentation (labelled)
        for i in range(steps_per_epoch_labelled):
            training_samples_l, GT_samples = training_generator_labelled[i]
            seg_loss = classifier.train_on_batch(training_samples_l, GT_samples)
            summary = tf.Summary(value=[tf.Summary.Value(tag="classifier_loss", simple_value = seg_loss[0]), ])
            writer.add_summary(summary, e*steps_per_epoch_labelled + i)
            summary = tf.Summary(value=[tf.Summary.Value(tag="classifier_acc", simple_value = seg_loss[1]), ])
            writer.add_summary(summary, e*steps_per_epoch_labelled + i)

        # train attention (paired)
        for i in range(steps_per_epoch_paired):
            training_samples, GT_samples, training_samples_u = training_generator_labelled_unlabelled[i]
            atten_loss = attention.train_on_batch([training_samples, GT_samples, training_samples_u], None)   
            summary = tf.Summary(value=[tf.Summary.Value(tag="attention_loss", simple_value = atten_loss), ])
            writer.add_summary(summary, e*steps_per_epoch_paired + i)

        
        print("epoch_end:", e, "seg loss:", seg_loss, "atten_loss:", atten_loss)
        
        attention.save("weights//" + "attention_" + str(e) + ".hdf")  
        classifier.save("weights//" + "classifier_" + str(e) + ".hdf") 
        
        training_generator_labelled_unlabelled.on_epoch_end()
        training_generator_labelled.on_epoch_end()

    return writer
    