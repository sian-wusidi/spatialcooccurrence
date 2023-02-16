#!/usr/bin/env python3

import numpy as np
import tensorflow as tf


def calculate_entropy(inputs):
    entropy = - inputs * np.log(inputs + 0.1)
    return entropy
    
def fit_models(classifier, attention, discriminator, combined, training_generator_labelled_unlabelled, training_generator_labelled, training_generator_unlabelled, batch_size, epochs, num_labelled_images, num_unlabelled_images, num_paired_images,  writer = None):    
    steps_per_epoch_labelled = num_labelled_images // batch_size
    steps_per_epoch_unlabelled = num_unlabelled_images // batch_size
    steps_per_epoch_paired = num_paired_images // batch_size
    valid = np.ones((batch_size, 1)) 
    fake =  np.zeros((batch_size, 1)) 
    seg_loss = 0
    atten_loss = 0
    combine_loss = 0
    d_loss_fake = 0
    d_loss_real = 0
    
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

        # train discriminator and adversarial training (unlabelled)
        for i in range(steps_per_epoch_unlabelled):
            training_samples_u, _ = training_generator_unlabelled[i]
            training_samples_l, GT_samples = training_generator_labelled[i]
            prediction = classifier.predict(training_samples_l) 
            prediction_u = classifier.predict(training_samples_u) 
            d_loss_real = discriminator.train_on_batch(prediction, valid) 
            d_loss_fake = discriminator.train_on_batch(prediction_u, fake)

            summary = tf.Summary(value=[tf.Summary.Value(tag="d_loss_real", simple_value = d_loss_real[0]), ])
            writer.add_summary(summary, e*steps_per_epoch_unlabelled + i)
            summary = tf.Summary(value=[tf.Summary.Value(tag="d_acc_real", simple_value = d_loss_real[1]), ])
            writer.add_summary(summary, e*steps_per_epoch_unlabelled + i)
            
            summary = tf.Summary(value=[tf.Summary.Value(tag="d_loss_fake", simple_value = d_loss_fake[0]), ])
            writer.add_summary(summary, e*steps_per_epoch_unlabelled + i)
            summary = tf.Summary(value=[tf.Summary.Value(tag="d_acc_fake", simple_value = d_loss_fake[1]), ])
            writer.add_summary(summary, e*steps_per_epoch_unlabelled + i)

            # adversarial training only when stablized 
            if d_loss_real[1] > 0.6 and d_loss_fake[1] > 0.6 and seg_loss[0] <= 0.3:  
                combine_loss = combined.train_on_batch(training_samples_u, valid, 0.001*np.squeeze(valid))  # weight 0.001
                summary = tf.Summary(value=[tf.Summary.Value(tag="adversarial loss", simple_value = combine_loss[0]), ])
                writer.add_summary(summary, e*steps_per_epoch_unlabelled + i)
                summary = tf.Summary(value=[tf.Summary.Value(tag="adversarial acc", simple_value = combine_loss[1]), ])
                writer.add_summary(summary, e*steps_per_epoch_unlabelled + i)
            else:
                combine_loss = 0      

        print("epoch_end:", e, "seg_loss:", seg_loss,"d_loss_real:", d_loss_real, "d_loss_fake:", d_loss_fake,  "atten_loss:", atten_loss, "combine_loss: ", combine_loss)

        attention.save("weights//" + "attention_" + str(e) + ".hdf")  
        classifier.save("weights//" + "classifier_" + str(e) + ".hdf") 
        discriminator.save("weights//" + "discriminator_" + str(e) + ".hdf") 
   
        training_generator_labelled_unlabelled.on_epoch_end()
        training_generator_labelled.on_epoch_end()
        training_generator_unlabelled.on_epoch_end()

    return writer
