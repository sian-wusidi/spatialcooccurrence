#!/usr/bin/env python3

import numpy as np

from keras import backend as K
from keras.models import Sequential, Model
from keras.layers import Input, Conv2D, BatchNormalization, Dense, Conv2DTranspose, Flatten, \
    Lambda, LeakyReLU, Activation, MaxPooling2D, Dropout, AveragePooling2D, DepthwiseConv2D, Concatenate, \
    UpSampling2D
from keras.regularizers import l2

from .losses import Buffer_DiceLoss, priorloss
from keras.engine import Layer, InputSpec
from keras.utils import conv_utils
import tensorflow as tf

def create_models(n_channels=3, wdecay=1e-5, bn_mom=0.9, bn_eps=1e-6):
    width = 256
    image_shape = (width, width, n_channels)
    GT_shape = (width, width, 4)
    latent_shape = (16, 16, 256)
    n_discriminator = 512
    leaky_relu_alpha = 0.2
    num_filters = 16
    OS = 16
    atrous_rates = (3, 6, 9)

    def create_classifier():

        inputs = Input(shape=image_shape, name="class_input")

        # 128   input - [batchsize,128,128,4] 
        c1 = Conv2D(num_filters, (3, 3), kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(wdecay))(inputs)
        c1 = BatchNormalization()(c1)
        c1 = Activation('elu')(c1)
        c1 = Dropout(0.1)(c1)
        c1 = Conv2D(num_filters, (3, 3), kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(wdecay))(c1)
        c1 = BatchNormalization()(c1)
        c1 = Activation('elu')(c1)
        p1 = MaxPooling2D((2, 2))(c1)
        
        # 64
        c2 = Conv2D(num_filters * 2, (3, 3), kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(wdecay))(p1)
        c2 = BatchNormalization()(c2)
        c2 = Activation('elu')(c2)
        c2 = Dropout(0.1)(c2)
        c2 = Conv2D(num_filters * 2, (3, 3), kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(wdecay))(c2)
        c2 = BatchNormalization()(c2)
        c2 = Activation('elu')(c2)
        p2 = MaxPooling2D((2, 2))(c2)
        
        # 32
        c3 = Conv2D(num_filters * 4, (3, 3), kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(wdecay))(p2)
        c3 = BatchNormalization()(c3)
        c3 = Activation('elu')(c3)
        c3 = Dropout(0.2)(c3)
        c3 = Conv2D(num_filters * 4, (3, 3), kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(wdecay))(c3)
        c3 = BatchNormalization()(c3)
        c3 = Activation('elu')(c3)
        p3 = MaxPooling2D((2, 2))(c3)
        
        # 16
        c4 = Conv2D(num_filters * 8, (3, 3), kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(wdecay))(p3)
        c4 = BatchNormalization()(c4)
        c4 = Activation('elu')(c4)
        c4 = Dropout(0.2)(c4)
        c4 = Conv2D(num_filters * 8, (3, 3), kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(wdecay))(c4)
        c4 = BatchNormalization()(c4)
        c4 = Activation('elu')(c4)
        p4 = MaxPooling2D(pool_size=(2, 2))(c4)
        
        # 8
        c5 = Conv2D(num_filters * 16, (3, 3), kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(wdecay))(p4)
        c5 = BatchNormalization()(c5)
        c5 = Activation('elu')(c5)
        c5 = Dropout(0.3)(c5)
        c5 = Conv2D(num_filters * 16, (3, 3), kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(wdecay))(c5)
        c5 = BatchNormalization()(c5)
        c5 = Activation('elu')(c5)
        
        # ASPP
        # Image Feature branch
        #out_shape = int(np.ceil(input_shape[0] / OS))
        b4 = AveragePooling2D(pool_size=(int(np.ceil(image_shape[0] / OS)), int(np.ceil(image_shape[1] / OS))))(c5)
        b4 = Conv2D(256, (1, 1), padding='same', use_bias=False)(b4)
        b4 = BatchNormalization(epsilon=1e-5)(b4)
        b4 = Activation('elu')(b4)
        b4 = BilinearUpsampling((int(np.ceil(image_shape[0] / OS)), int(np.ceil(image_shape[1] / OS))))(b4)

        # simple 1x1
        b0 = Conv2D(256, (1, 1), padding='same', use_bias=False)(c5)
        b0 = BatchNormalization(epsilon=1e-5)(b0)
        b0 = Activation('elu')(b0)
        # rate = 3 (6)
        b1 = SepConv_BN(c5, 256, rate=atrous_rates[0], depth_activation=False, epsilon=1e-5)
        # rate = 6 (12)
        b2 = SepConv_BN(c5, 256, rate=atrous_rates[1], depth_activation=False, epsilon=1e-5)
        # rate = 9 (18)
        b3 = SepConv_BN(c5, 256, rate=atrous_rates[2], depth_activation=False, epsilon=1e-5)

        # concatenate ASPP branches & project
        c5 = Concatenate()([b4, b0, b1, b2, b3])
        
        # simple 1x1 again
        c5 = Conv2D(256, (1, 1), padding='same', use_bias=False)(c5) 


        c5 = BatchNormalization(epsilon=1e-5)(c5)
        c5 = Activation('elu')(c5)
        c5 = Dropout(0.1)(c5)

        # 16
        u6 = Conv2DTranspose(num_filters * 8, (2, 2), strides=(2, 2), padding='same', kernel_regularizer=l2(wdecay))(c5)
        u6 = Concatenate()([u6, c4])
        c6 = Conv2D(num_filters * 8, (3, 3), kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(wdecay))(u6)
        c6 = BatchNormalization()(c6)
        c6 = Activation('elu')(c6)
        c6 = Dropout(0.2)(c6)
        c6 = Conv2D(num_filters * 8, (3, 3), kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(wdecay))(c6)
        c6 = BatchNormalization()(c6)
        c6 = Activation('elu')(c6)

        # 32
        u7 = Conv2DTranspose(num_filters * 4, (2, 2), strides=(2, 2), padding='same', kernel_regularizer=l2(wdecay))(c6)
        u7 = Concatenate()([u7, c3])
        c7 = Conv2D(num_filters * 4, (3, 3), kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(wdecay))(u7)
        c7 = BatchNormalization()(c7)
        c7 = Activation('elu')(c7)
        c7 = Dropout(0.2)(c7)
        c7 = Conv2D(num_filters * 4, (3, 3), kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(wdecay))(c7)
        c7 = BatchNormalization()(c7)
        c7 = Activation('elu')(c7)
        
        # 64
        u8 = Conv2DTranspose(num_filters * 2, (2, 2), strides=(2, 2), padding='same')(c7)
        u8 = Concatenate()([u8, c2])
        c8 = Conv2D(num_filters * 2, (3, 3), kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(wdecay))(u8)
        c8 = BatchNormalization()(c8)
        c8 = Activation('elu')(c8)
        c8 = Dropout(0.1)(c8)
        c8 = Conv2D(num_filters * 2, (3, 3), kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(wdecay))(c8)
        c8 = BatchNormalization()(c8)
        c8 = Activation('elu')(c8)
        
        # 128
        u9 = Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding='same')(c8)
        u9 = Concatenate()([u9, c1])
        c9 = Conv2D(num_filters, (3, 3), kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(wdecay))(u9)
        c9 = BatchNormalization()(c9)
        c9 = Activation('elu')(c9)
        c9 = Dropout(0.1)(c9)
        c9 = Conv2D(num_filters, (3, 3), kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(wdecay))(c9)
        c9 = BatchNormalization()(c9)
        c9 = Activation('elu')(c9)
        
        finalConv = Conv2D(4, (1, 1), activation='sigmoid')(c9)
        
        model = Model(inputs = inputs, outputs = finalConv, name = "classifier")
        intermediate = Model(inputs = inputs, outputs = [c5, c4, c3, c2, c1], name = "classifier_intermediate")
        return model, intermediate

    def conv_block_p(x, filters, leaky=True, transpose=False, name='', dropout = 0.1):
        conv = Conv2DTranspose if transpose else Conv2D
        activation = LeakyReLU(leaky_relu_alpha) if leaky else Activation('relu')
        layers = [
            conv(filters, 5, strides=2, padding='same', name=name + 'conv'),
            BatchNormalization(momentum=bn_mom, epsilon=bn_eps, name=name + 'bn'),
            activation,
            Dropout(dropout)
        ]
        if x is None:
            return layers
        for layer in layers:
            x = layer(x)
        return x

    def create_discriminator_depth5():
        x = Input(shape=GT_shape, name='dis_input')

        layers = [
            Conv2D(64, 5, padding='same', kernel_regularizer=l2(wdecay), kernel_initializer='he_uniform', name='dis_blk_1_conv'),
            BatchNormalization(),
            LeakyReLU(leaky_relu_alpha),
            MaxPooling2D((4,4)),
            *conv_block_p(None, 128, leaky=True, name='dis_blk_2_'),
            *conv_block_p(None, 128, leaky=True, name='dis_blk_3_'),
            *conv_block_p(None, 256, leaky=True, name='dis_blk_4_'),
            *conv_block_p(None, 256, leaky=True, name='dis_blk_5_'),
            Flatten(),
            Dense(n_discriminator, name='dis_dense'),
            BatchNormalization(name='dis_bn'),
            LeakyReLU(leaky_relu_alpha),
            Dense(1, activation='sigmoid', name='dis_output') #activation='sigmoid'
        ]

        y = x
        y_feat = None

        for i, layer in enumerate(layers, 1):
            y = layer(y)
        return Model(x, y, name='discriminator')
        
    classifier, intermediate = create_classifier()
    discriminator = create_discriminator_depth5()
    return classifier, intermediate, discriminator


def build_graph(classifier, intermediate, buffersize = 16):
    num_filters = 16
    wdecay = 1e-5
    image_shape = K.int_shape(classifier.input)[1:]
    GT_shape = (K.int_shape(classifier.input)[1], K.int_shape(classifier.input)[2],4)
    x = Input(shape=image_shape, name='input_image_labelled')
    x1 = Input(shape=image_shape, name='input_image_unlabelled')
    GT = Input(shape=GT_shape, name='input_image_GT')
    c5, c4, c3, c2, c1 = intermediate(x)
    c5_1, c4_1, c3_1, c2_1, c1_1 = intermediate(x1)
    x_prediction1 = classifier(x1)
    
    c5_con = Concatenate()([c5, c5_1])
    u6 = Conv2DTranspose(num_filters * 8, (2, 2), strides=(2, 2), padding='same', kernel_regularizer=l2(wdecay))(c5_con)
    u6 = Concatenate()([u6, c4, c4_1])
    c6 = Conv2D(num_filters * 8, (3, 3), kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(wdecay))(u6)
    c6 = BatchNormalization()(c6)
    c6 = Activation('elu')(c6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(num_filters * 8, (3, 3), kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(wdecay))(c6)
    c6 = BatchNormalization()(c6)
    c6 = Activation('elu')(c6)

    # 32
    u7 = Conv2DTranspose(num_filters * 4, (2, 2), strides=(2, 2), padding='same', kernel_regularizer=l2(wdecay))(c6)
    u7 = Concatenate()([u7, c3, c3_1])
    c7 = Conv2D(num_filters * 4, (3, 3), kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(wdecay))(u7)
    c7 = BatchNormalization()(c7)
    c7 = Activation('elu')(c7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(num_filters * 4, (3, 3), kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(wdecay))(c7)
    c7 = BatchNormalization()(c7)
    c7 = Activation('elu')(c7)
    
    # 64
    u8 = Conv2DTranspose(num_filters * 2, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = Concatenate()([u8, c2, c2_1])
    c8 = Conv2D(num_filters * 2, (3, 3), kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(wdecay))(u8)
    c8 = BatchNormalization()(c8)
    c8 = Activation('elu')(c8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv2D(num_filters * 2, (3, 3), kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(wdecay))(c8)
    c8 = BatchNormalization()(c8)
    c8 = Activation('elu')(c8)
    
    # 128
    u9 = Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = Concatenate()([u9, c1,c1_1])
    c9 = Conv2D(num_filters, (3, 3), kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(wdecay))(u9)
    c9 = BatchNormalization()(c9)
    c9 = Activation('elu')(c9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv2D(num_filters, (3, 3), kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(wdecay))(c9)
    c9 = BatchNormalization()(c9)
    c9 = Activation('elu')(c9)
    mask = Conv2D(4, (1, 1), activation='sigmoid')(c9)    
    
    # buffered dice loss
    attention = Model([x, GT, x1], mask, name='attention')
    GT_p = MaxPooling2D((buffersize,buffersize))(GT)
    GT_buffer = UpSampling2D((buffersize,buffersize))(GT_p)
    masked_loss = Buffer_DiceLoss(mask*x_prediction1, mask*GT, mask*GT_buffer) 
    attention.add_loss(masked_loss)
    
    # regularization loss of mask
    reg_loss = 1 - 0.2*K.mean(mask[:,:,:,0]) - 0.2*K.mean(mask[:,:,:,1]) - 0.4*K.mean(mask[:,:,:,2]) - 0.2*K.mean(mask[:,:,:,3]) 

    attention.add_loss(reg_loss)
    
    return attention

    
def build_graph_all(classifier, intermediate, discriminator, buffersize = 16):
    num_filters = 16
    wdecay = 1e-5
    image_shape = K.int_shape(classifier.input)[1:]
    GT_shape = (K.int_shape(classifier.input)[1], K.int_shape(classifier.input)[2],4)
    x = Input(shape=image_shape, name='input_image_labelled')
    x1 = Input(shape=image_shape, name='input_image_unlabelled')
    GT = Input(shape=GT_shape, name='input_image_GT')
    c5, c4, c3, c2, c1 = intermediate(x)
    c5_1, c4_1, c3_1, c2_1, c1_1 = intermediate(x1)
    x_prediction1 = classifier(x1)
    
    c5_con = Concatenate()([c5, c5_1])
    u6 = Conv2DTranspose(num_filters * 8, (2, 2), strides=(2, 2), padding='same', kernel_regularizer=l2(wdecay))(c5_con)
    u6 = Concatenate()([u6, c4, c4_1])
    c6 = Conv2D(num_filters * 8, (3, 3), kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(wdecay))(u6)
    c6 = BatchNormalization()(c6)
    c6 = Activation('elu')(c6)
    c6 = Dropout(0.2)(c6)
    c6 = Conv2D(num_filters * 8, (3, 3), kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(wdecay))(c6)
    c6 = BatchNormalization()(c6)
    c6 = Activation('elu')(c6)

    # 32
    u7 = Conv2DTranspose(num_filters * 4, (2, 2), strides=(2, 2), padding='same', kernel_regularizer=l2(wdecay))(c6)
    u7 = Concatenate()([u7, c3, c3_1])
    c7 = Conv2D(num_filters * 4, (3, 3), kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(wdecay))(u7)
    c7 = BatchNormalization()(c7)
    c7 = Activation('elu')(c7)
    c7 = Dropout(0.2)(c7)
    c7 = Conv2D(num_filters * 4, (3, 3), kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(wdecay))(c7)
    c7 = BatchNormalization()(c7)
    c7 = Activation('elu')(c7)
    
    # 64
    u8 = Conv2DTranspose(num_filters * 2, (2, 2), strides=(2, 2), padding='same')(c7)
    u8 = Concatenate()([u8, c2, c2_1])
    c8 = Conv2D(num_filters * 2, (3, 3), kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(wdecay))(u8)
    c8 = BatchNormalization()(c8)
    c8 = Activation('elu')(c8)
    c8 = Dropout(0.1)(c8)
    c8 = Conv2D(num_filters * 2, (3, 3), kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(wdecay))(c8)
    c8 = BatchNormalization()(c8)
    c8 = Activation('elu')(c8)
    
    # 128
    u9 = Conv2DTranspose(num_filters, (2, 2), strides=(2, 2), padding='same')(c8)
    u9 = Concatenate()([u9, c1,c1_1])
    c9 = Conv2D(num_filters, (3, 3), kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(wdecay))(u9)
    c9 = BatchNormalization()(c9)
    c9 = Activation('elu')(c9)
    c9 = Dropout(0.1)(c9)
    c9 = Conv2D(num_filters, (3, 3), kernel_initializer='he_normal', padding='same', kernel_regularizer=l2(wdecay))(c9)
    c9 = BatchNormalization()(c9)
    c9 = Activation('elu')(c9)
    mask = Conv2D(4, (1, 1), activation='sigmoid')(c9)    
    attention = Model([x, GT, x1], mask, name='attention')

    # buffered dice loss
    GT_p = MaxPooling2D((buffersize, buffersize))(GT)
    GT_buffer = UpSampling2D((buffersize,buffersize))(GT_p)
    masked_loss = Buffer_DiceLoss(mask*x_prediction1, mask*GT, mask*GT_buffer)
    attention.add_loss(masked_loss)
    
    # regularization loss of mask
    reg_loss = 1 - 0.2*K.mean(mask[:,:,:,0]) - 0.2*K.mean(mask[:,:,:,1]) - 0.4*K.mean(mask[:,:,:,2]) - 0.2*K.mean(mask[:,:,:,3]) # for KOMB
    attention.add_loss(reg_loss)
    
    x_prediction1_entropy = Lambda(lambda x: -x*K.log(x+0.1))(x_prediction1)
    dis_x1_prediction = discriminator(x_prediction1_entropy)
    combined = Model(x1, dis_x1_prediction, name='comb')

    # prior loss
    prior = priorloss(x_prediction1)
    combined.add_loss(prior)

    return attention, combined         


def SepConv_BN(x, filters, stride=1, kernel_size=3, rate=1, depth_activation=False, epsilon=1e-3):
    """ SepConv with BN between depthwise & pointwise. Optionally add activation after BN
        Implements right "same" padding for even kernel sizes
        Args:
            x: input tensor
            filters: num of filters in pointwise convolution
            prefix: prefix before name
            stride: stride at depthwise conv
            kernel_size: kernel size for depthwise convolution
            rate: atrous rate for depthwise convolution
            depth_activation: flag to use activation between depthwise & poinwise convs
            epsilon: epsilon to use in BN layer
            the code is based on keras implementation of deeplabV3+ https://github.com/bonlime/keras-deeplab-v3-plus
    """

    if stride == 1:
        depth_padding = 'same'
    else:
        kernel_size_effective = kernel_size + (kernel_size - 1) * (rate - 1)
        pad_total = kernel_size_effective - 1
        pad_beg = pad_total // 2
        pad_end = pad_total - pad_beg
        x = ZeroPadding2D((pad_beg, pad_end))(x)
        depth_padding = 'valid'

    if not depth_activation:
        x = Activation('elu')(x)
        
    x = DepthwiseConv2D((kernel_size, kernel_size), strides=(stride, stride), dilation_rate=(rate, rate), padding=depth_padding, use_bias=False)(x) #depthwise
    x = BatchNormalization(epsilon=epsilon)(x)
    
    if depth_activation:
        x = Activation('elu')(x)
        
    x = Conv2D(filters, (1, 1), padding='same', use_bias=False)(x) #pointwise
    x = BatchNormalization(epsilon=epsilon)(x)
    
    if depth_activation:
        x = Activation('elu')(x)

    return x


class BilinearUpsampling(Layer):
    """Just a simple bilinear upsampling layer. Works only with TF.
       Args:
           upsampling: tuple of 2 numbers > 0. The upsampling ratio for h and w
           output_size: used instead of upsampling arg if passed!
    """

    def __init__(self, upsampling=(2, 2), output_size=None, data_format=None, **kwargs):

        super(BilinearUpsampling, self).__init__(**kwargs)

        self.data_format = K.image_data_format()
        self.input_spec = InputSpec(ndim=4)
        if output_size:
            self.output_size = conv_utils.normalize_tuple(output_size, 2, 'output_size')
            self.upsampling = None
        else:
            self.output_size = None
            self.upsampling = conv_utils.normalize_tuple(upsampling, 2, 'upsampling')

    def compute_output_shape(self, input_shape):
        if self.upsampling:
            height = self.upsampling[0] * \
                input_shape[1] if input_shape[1] is not None else None
            width = self.upsampling[1] * \
                input_shape[2] if input_shape[2] is not None else None
        else:
            height = self.output_size[0]
            width = self.output_size[1]
        return (input_shape[0],
                height,
                width,
                input_shape[3])

    def call(self, inputs):
        if self.upsampling:
            return K.tf.image.resize_bilinear(inputs, (inputs.shape[1] * self.upsampling[0], inputs.shape[2] * self.upsampling[1]), align_corners=True)
            #return K.tensorflow_backend.tf.image.resize(inputs, (inputs.shape[1] * self.upsampling[0], inputs.shape[2] * self.upsampling[1]))
        else:
            return K.tf.image.resize_bilinear(inputs, (self.output_size[0], self.output_size[1]), align_corners=True)
            #return K.tensorflow_backend.tfimage.resize(inputs, (self.output_size[0], self.output_size[1]))
    def get_config(self):
        config = {'upsampling': self.upsampling,
                  'output_size': self.output_size,
                  'data_format': self.data_format}
        base_config = super(BilinearUpsampling, self).get_config()
        return dict(list(base_config.items()) + list(config.items()))
        

def get_pixel_value(img, x, y):
    """
    https://github.com/kevinzakka/spatial-transformer-network/
    Utility function to get pixel value for coordinate
    vectors x and y from a  4D tensor image.
    Input
    -----
    - img: tensor of shape (B, H, W, C)
    - x: flattened tensor of shape (B*H*W,)
    - y: flattened tensor of shape (B*H*W,)
    Returns
    -------
    - output: tensor of shape (B, H, W, C)
    """
    shape = tf.shape(x)
    batch_size = shape[0]
    height = shape[1]
    width = shape[2]

    batch_idx = tf.range(0, batch_size)
    batch_idx = tf.reshape(batch_idx, (batch_size, 1, 1))
    b = tf.tile(batch_idx, (1, height, width))

    indices = tf.stack([b, y, x], 3)

    return tf.gather_nd(img, indices)


def bilinear_sampler(img, grid):
    """
    adapt from:
    https://github.com/kevinzakka/spatial-transformer-network/
    Performs bilinear sampling of the input images according to the
    normalized coordinates provided by the sampling grid. Note that
    the sampling is done identically for each channel of the input.
    To test if the function works properly, output image should be
    identical to input image when theta is initialized to identity
    transform.
    Input
    -----
    - img: batch of images in (B, H, W, C) layout.
    - grid: x, y which is the output of affine_grid_generator.
    Returns
    -------
    - out: interpolated images according to grids. Same size as grid.
    """
    H = tf.shape(img)[1]
    W = tf.shape(img)[2]
    max_y = tf.cast(H - 1, 'int32')
    max_x = tf.cast(W - 1, 'int32')
    zero = tf.zeros([], dtype='int32')
    x = grid[:,:,:,0] #[-1,1]
    y = grid[:,:,:,1]

    # rescale x and y to [0, W-1/H-1]
    x = tf.cast(x, 'float32')
    y = tf.cast(y, 'float32')
    x = 0.5 * ((x + 1.0) * tf.cast(max_x-1, 'float32'))
    y = 0.5 * ((y + 1.0) * tf.cast(max_y-1, 'float32'))

    # grab 4 nearest corner points for each (x_i, y_i)
    x0 = tf.cast(tf.floor(x), 'int32') # B, W, H
    x1 = x0 + 1
    y0 = tf.cast(tf.floor(y), 'int32')
    y1 = y0 + 1

    # clip to range [0, H-1/W-1] to not violate img boundaries
    x0 = tf.clip_by_value(x0, zero, max_x)
    x1 = tf.clip_by_value(x1, zero, max_x)
    y0 = tf.clip_by_value(y0, zero, max_y)
    y1 = tf.clip_by_value(y1, zero, max_y)

    # get pixel value at corner coords
    Ia = get_pixel_value(img, x0, y0) #B,H,W,3
    Ib = get_pixel_value(img, x0, y1)
    Ic = get_pixel_value(img, x1, y0)
    Id = get_pixel_value(img, x1, y1)

    # recast as float for delta calculation
    x0 = tf.cast(x0, 'float32')
    x1 = tf.cast(x1, 'float32')
    y0 = tf.cast(y0, 'float32')
    y1 = tf.cast(y1, 'float32')

    # calculate deltas
    wa = (x1-x) * (y1-y)
    wb = (x1-x) * (y-y0)
    wc = (x-x0) * (y1-y)
    wd = (x-x0) * (y-y0)

    # add dimension for addition
    wa = tf.expand_dims(wa, axis=3)
    wb = tf.expand_dims(wb, axis=3)
    wc = tf.expand_dims(wc, axis=3)
    wd = tf.expand_dims(wd, axis=3)

    # compute output
    out = tf.add_n([wa*Ia, wb*Ib, wc*Ic, wd*Id])

    return out