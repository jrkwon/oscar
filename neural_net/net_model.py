#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 13:23:14 2017
History:
11/28/2020: modified for OSCAR 

@author: jaerock
@author: ninad#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""

from keras.models import Sequential, Model
from keras.layers import Lambda, Dropout, Flatten, Dense, Activation, concatenate
from keras.layers import Conv2D, Convolution2D, MaxPooling2D, BatchNormalization, Input
from keras import losses, optimizers
import keras.backend as K
import tensorflow as tf

import os
import const
from config import Config

config = Config.neural_net
config_rn = Config.run_neural

def model_ce491():
    input_shape = (config['input_image_height'],
                    config['input_image_width'],
                    config['input_image_depth'])

    return Sequential([
        Lambda(lambda x: x/127.5 - 1.0, input_shape=input_shape),
        Conv2D(24, (5, 5), strides=(2,2), activation='relu', name='conv2d_1'),
        Conv2D(36, (5, 5), strides=(2,2), activation='relu', name='conv2d_2'),
        Conv2D(48, (5, 5), strides=(2,2), activation='relu', name='conv2d_3'),
        Conv2D(64, (3, 3), activation='relu', name='conv2d_4'),
        Conv2D(64, (3, 3), activation='relu', name='conv2d_last'),
        Flatten(),
        Dense(100, activation='relu', name='fc_1'),
        Dense(50, activation='relu', name='fc_2'),
        Dense(10, activation='relu', name='fc_3'),
        Dense(config['num_outputs'], name='fc_str')])

def model_jaerock():
    input_shape = (config['input_image_height'],
                    config['input_image_width'],
                    config['input_image_depth'])

    return Sequential([
        Lambda(lambda x: x/127.5 - 1.0, input_shape=input_shape),
        Conv2D(24, (5, 5), strides=(2,2)),
        Conv2D(36, (5, 5), strides=(2,2)),
        Conv2D(48, (5, 5), strides=(2,2)),
        Conv2D(64, (3, 3)),
        Conv2D(64, (3, 3), name='conv2d_last'),
        Flatten(),
        Dense(1000),
        Dense(100),
        Dense(50),
        Dense(10),
        Dense(config['num_outputs'])], name='fc_str')
    
def model_jaerock_vel():
    img_shape = (config['input_image_height'],
                    config['input_image_width'],
                    config['input_image_depth'],)
    vel_shape = 1
    ######img model#######
    img_input = Input(shape=img_shape)
    lamb = Lambda(lambda x: x/127.5 - 1.0)(img_input)
    conv_1 = Conv2D(24, (5, 5), strides=(2,2))(lamb)
    conv_2 = Conv2D(36, (5, 5), strides=(2,2))(conv_1)
    conv_3 = Conv2D(48, (5, 5), strides=(2,2))(conv_2)
    conv_4 = Conv2D(64, (3, 3))(conv_3)
    conv_5 = Conv2D(64, (3, 3), name='conv2d_last')(conv_4)
    flat = Flatten()(conv_5)
    fc_1 = Dense(1000, name='fc_1')(flat)
    fc_2 = Dense(100, name='fc_2')(fc_1)
    
    ######vel model#######
    vel_input = Input(shape=[vel_shape])
    fc_vel = Dense(50, name='fc_vel')(vel_input)
    
    ######concat##########
    concat_img_vel = concatenate([fc_2, fc_vel])
    fc_3 = Dense(50, name='fc_3')(concat_img_vel)
    fc_4 = Dense(10, name='fc_4')(fc_3)
    fc_last = Dense(2, name='fc_str')(fc_4)
    
    model = Model(inputs=[img_input, vel_input], output=fc_last)

    return model

def model_jaerock_elu():
    input_shape = (config['input_image_height'],
                    config['input_image_width'],
                    config['input_image_depth'])

    return Sequential([
        Lambda(lambda x: x/127.5 - 1.0, input_shape=input_shape),
        Conv2D(24, (5, 5), strides=(2,2), activation='elu'),
        Conv2D(36, (5, 5), strides=(2,2), activation='elu'),
        Conv2D(48, (5, 5), strides=(2,2), activation='elu'),
        Conv2D(64, (3, 3), activation='elu'),
        Conv2D(64, (3, 3), activation='elu', name='conv2d_last'),
        Flatten(),
        Dense(1000, activation='elu'),
        Dense(100, activation='elu'),
        Dense(50, activation='elu'),
        Dense(10, activation='elu'),
        Dense(config['num_outputs'])], name='fc_str')

def model_donghyun(): #jaerock과 동일하지만 필터사이즈를 초반에는 크게
    img_shape = (config['input_image_height'],
                    config['input_image_width'],
                    config['input_image_depth'],)
    
    ######img model#######
    img_input = Input(shape=img_shape)
    lamb = Lambda(lambda x: x/127.5 - 1.0)(img_input)
    conv_1 = Conv2D(24, (8, 8), strides=(2,2), activation='elu')(lamb)
    conv_2 = Conv2D(36, (6, 6), strides=(2,2), activation='elu')(conv_1)
    conv_3 = Conv2D(48, (5, 5), strides=(2,2), activation='elu')(conv_2)
    conv_4 = Conv2D(64, (3, 3), activation='elu')(conv_3)
    conv_5 = Conv2D(64, (3, 3), name='conv2d_last', activation='elu')(conv_4)
    flat = Flatten()(conv_5)
    fc_1 = Dense(1000, activation='elu', name='fc_1')(flat)
    fc_2 = Dense(100,  activation='elu', name='fc_2')(fc_1)
    fc_3 = Dense(50,   activation='elu', name='fc_3')(fc_2)
    fc_4 = Dense(10,   activation='elu', name='fc_4')(fc_3)
    fc_last = Dense(config['num_outputs'], name='fc_str')(fc_4)
    
    model = Model(inputs=img_input, output=fc_last)

    return model
    
def model_donghyun2(): #donghyun과 동일하지만 큰 필터개수를 더 많게
    img_shape = (config['input_image_height'],
                    config['input_image_width'],
                    config['input_image_depth'],)
    
    ######img model#######
    img_input = Input(shape=img_shape)
    lamb = Lambda(lambda x: x/127.5 - 1.0)(img_input)
    conv_1 = Conv2D(64, (8, 8), strides=(2,2), activation='elu')(lamb)
    conv_2 = Conv2D(48, (6, 6), strides=(2,2), activation='elu')(conv_1)
    conv_3 = Conv2D(36, (5, 5), strides=(2,2), activation='elu')(conv_2)
    conv_4 = Conv2D(24, (3, 3), activation='elu')(conv_3)
    conv_5 = Conv2D(24, (3, 3), activation='elu', name='conv2d_last')(conv_4)
    flat = Flatten()(conv_5)
    fc_1 = Dense(1000, activation='elu', name='fc_1')(flat)
    fc_2 = Dense(100,  activation='elu', name='fc_2')(fc_1)
    fc_3 = Dense(50,   activation='elu', name='fc_3')(fc_2)
    fc_4 = Dense(10,   activation='elu', name='fc_4')(fc_3)
    fc_last = Dense(1, name='fc_str')(fc_4)
    
    model = Model(inputs=img_input, output=fc_last)

    return model

def model_donghyun3(): #필터사이즈를 donghyun2에 비해 전체적으로 더 크게
    img_shape = (config['input_image_height'],
                    config['input_image_width'],
                    config['input_image_depth'],)
    
    ######img model#######
    img_input = Input(shape=img_shape)
    lamb = Lambda(lambda x: x/127.5 - 1.0)(img_input)
    conv_1 = Conv2D(64, (8, 8), strides=(2,2), activation='elu')(lamb)
    conv_2 = Conv2D(48, (8, 8), strides=(2,2), activation='elu')(conv_1)
    conv_3 = Conv2D(36, (5, 5), activation='elu')(conv_2)
    conv_4 = Conv2D(24, (5, 5), activation='elu')(conv_3)
    conv_5 = Conv2D(24, (5, 5), activation='elu', name='conv2d_last')(conv_4)
    flat = Flatten()(conv_5)
    fc_1 = Dense(1000, activation='elu', name='fc_1')(flat)
    fc_2 = Dense(100,  activation='elu', name='fc_2')(fc_1)
    fc_3 = Dense(50,   activation='elu', name='fc_3')(fc_2)
    fc_4 = Dense(10,   activation='elu', name='fc_4')(fc_3)
    fc_last = Dense(1, name='fc_str')(fc_4)
    
    model = Model(inputs=img_input, output=fc_last)

    return model

def model_donghyun4(): #레이어줄이기
    img_shape = (config['input_image_height'],
                    config['input_image_width'],
                    config['input_image_depth'],)
    
    ######img model#######
    img_input = Input(shape=img_shape)
    lamb = Lambda(lambda x: x/127.5 - 1.0)(img_input)
    conv_1 = Conv2D(64, (8, 8), strides=(2,2), activation='elu')(lamb)
    conv_2 = Conv2D(48, (8, 8), strides=(2,2), activation='elu')(conv_1)
    conv_3 = Conv2D(36, (5, 5), activation='elu')(conv_2)
    conv_4 = Conv2D(48, (5, 5), activation='elu')(conv_3)
    flat = Flatten()(conv_4)
    fc_1 = Dense(1000, activation='elu', name='fc_1')(flat)
    fc_2 = Dense(100,  activation='elu', name='fc_2')(fc_1)
    fc_3 = Dense(50,   activation='elu', name='fc_3')(fc_2)
    fc_4 = Dense(10,   activation='elu', name='fc_4')(fc_3)
    fc_last = Dense(1, name='fc_str')(fc_4)
    
    model = Model(inputs=img_input, output=fc_last)

    return model

def model_donghyun5(): # 모든 레이어들이 fc로 들어가도록
    img_shape = (config['input_image_height'],
                    config['input_image_width'],
                    config['input_image_depth'],)
    
    ######img model#######
    img_input = Input(shape=img_shape)
    lamb = Lambda(lambda x: x/127.5 - 1.0)(img_input)
    conv_1 = Conv2D(64, (8, 8), strides=(2,2), activation='elu')(lamb)
    conv_2 = Conv2D(48, (8, 8), strides=(2,2), activation='elu', name='conv2d_2')(conv_1)
    conv_3 = Conv2D(36, (5, 5), activation='elu', name='conv2d_3')(conv_2)
    conv_4 = Conv2D(24, (5, 5), activation='elu', name='conv2d_4')(conv_3)
    conv_5 = Conv2D(24, (5, 5), activation='elu', name='conv2d_last')(conv_4)
    flat_1 = Flatten()(conv_2)
    flat_2 = Flatten()(conv_3)
    flat_3 = Flatten()(conv_4)
    flat_4 = Flatten()(conv_5)
    concat = concatenate([flat_1, flat_2, flat_3, flat_4])
    fc_1 = Dense(1000, activation='elu', name='fc_1')(concat)
    fc_2 = Dense(100,  activation='elu', name='fc_2')(fc_1)
    fc_3 = Dense(50,   activation='elu', name='fc_3')(fc_2)
    fc_4 = Dense(10,   activation='elu', name='fc_4')(fc_3)
    fc_last = Dense(1, name='fc_str')(fc_4)
    
    model = Model(inputs=img_input, output=fc_last)

    return model

def model_donghyun6(): # resnet 처럼
    from keras.layers import add, Concatenate, ELU, UpSampling2D
    img_shape = (config['input_image_height'],
                    config['input_image_width'],
                    config['input_image_depth'],)
    
    ######img model#######
    img_input = Input(shape=img_shape)
    lamb = Lambda(lambda x: x/127.5 - 1.0)(img_input)
    conv_1  = Conv2D(64, (5, 5), strides=(2,2), name='conv_1')(lamb)
    conv_1_bn = BatchNormalization()(conv_1)
    conv_1_elu = ELU()(conv_1_bn)
    
    conv_2  = Conv2D(64, (5, 5), strides=(2,2), name='conv_2')(conv_1_elu)
    conv_2_bn = BatchNormalization()(conv_2)
    conv_2_elu = ELU()(conv_2_bn)
    
    conv_3_1= Conv2D(64, (5, 5), strides=(2,2), padding='same', name='conv_3_1')(conv_2_elu)
    conv_3_2= Conv2D(64, (3, 3), strides=(2,2), padding='same', name='conv_3_2')(conv_2_elu)
    conc_1  = Concatenate(axis=3)([conv_3_1, conv_3_2])
    conv_3_bn = BatchNormalization()(conc_1)
    conv_3_elu = ELU()(conv_3_bn)
    
    conv_3_2_bn = BatchNormalization()(conv_3_2)
    conv_3_2_elu = ELU()(conv_3_2_bn)
    
    conv_4_1= Conv2D(128, (3, 3), name='conv_4_1')(conv_3_elu)
    conv_4_2= Conv2D(64, (3, 3), name='conv_4_2')(conv_3_2_elu)
    conc_2  = Concatenate(axis=3)([conv_4_1, conv_4_2])
    conv_4_bn = BatchNormalization()(conc_2)
    conv_4_elu = ELU()(conv_4_bn)
    
    conv_4_2_bn = BatchNormalization()(conv_4_2)
    conv_4_2_elu = ELU()(conv_4_2_bn)
    
    conv_5_1= Conv2D(128, (3, 3), name='conv_5_1')(conv_4_elu)
    conv_5_2= Conv2D(64, (3, 3), name='conv_5_2')(conv_4_2_elu)
    conc_3  = Concatenate(axis=3)([conv_5_1, conv_5_2])
    
    conv_5_bn = BatchNormalization()(conc_3)
    conv_5_elu = ELU()(conv_5_bn)
    
    conv_6  = Conv2D(64, (3, 3), name='conv2d_last')(conv_5_elu)
    
    flat_1  = Flatten()(conv_6)
    
    fc_1 = Dense(100, activation='elu', name='fc_1')(flat_1)
    drop_1 = Dropout(0.2)(fc_1)
    fc_2 = Dense(50,  activation='elu', name='fc_2')(drop_1)
    drop_2 = Dropout(0.2)(fc_2)
    fc_3 = Dense(10,   activation='elu', name='fc_3')(drop_2)
    
    fc_last = Dense(1, name='fc_str')(fc_3)
    
    model = Model(inputs=img_input, output=fc_last)

    return model

def model_donghyun7(): # resnet 처럼
    from keras.layers import add, Concatenate, ELU, UpSampling2D
    img_shape = (config['input_image_height'],
                    config['input_image_width'],
                    config['input_image_depth'],)
    
    ######img model#######
    img_input = Input(shape=img_shape)
    lamb = Lambda(lambda x: x/127.5 - 1.0)(img_input)
    conv_1  = Conv2D(64, (8, 8), strides=(2,2), activation='elu', name='conv_1')(lamb)
    conv_2  = Conv2D(64, (6, 6), strides=(2,2), activation='elu', name='conv_2')(conv_1)
    conv_3_1= Conv2D(128, (5, 5), strides=(2,2), padding='same', activation='elu', name='conv_3_1')(conv_2)
    conv_3_2= Conv2D(128, (3, 3), strides=(2,2), padding='same', activation='elu', name='conv_3_2')(conv_2)
    conc_1  = Concatenate(axis=3)([conv_3_1, conv_3_2])
    
    conv_4_1= Conv2D(256, (3, 3), activation='elu')(conc_1)
    conv_4_2= Conv2D(256, (3, 3), activation='elu')(conv_3_2)
    conc_2  = Concatenate(axis=3)([conv_4_1, conv_4_2])
    
    conv_5_1= Conv2D(512, (3, 3), activation='elu', name='conv_5_1')(conc_2)
    conv_5_2= Conv2D(512, (3, 3), activation='elu', name='conv_5_2')(conv_4_2)
    conc_3  = Concatenate(axis=3)([conv_5_1, conv_5_2])
    conv_6  = Conv2D(512, (3, 3), activation='elu', name='conv2d_last')(conc_3)
    
    flat_1  = Flatten()(conv_6)
    fc_1 = Dense(1000, activation='elu', name='fc_1')(flat_1)
    fc_2 = Dense(100,  activation='elu', name='fc_2')(fc_1)
    fc_3 = Dense(50,   activation='elu', name='fc_3')(fc_2)
    fc_last = Dense(1, name='fc_str')(fc_3)
    
    model = Model(inputs=img_input, output=fc_last)

    return model

def model_donghyun8(): # resnet 처럼
    from keras.layers import Concatenate, ELU
    img_shape = (config['input_image_height'],
                    config['input_image_width'],
                    config['input_image_depth'],)
    
    ######img model#######
    img_input = Input(shape=img_shape)
    lamb = Lambda(lambda x: x/127.5 - 1.0)(img_input)
    conv_1 = Conv2D(24, (5, 5), name='conv_1')(lamb)
    conv_1_bn = BatchNormalization()(conv_1)
    conv_1_elu = ELU()(conv_1_bn)
    
    conv_1_pl = MaxPooling2D(pool_size=(2,2))(conv_1_elu)
    conv_1_pl2 = MaxPooling2D(pool_size=(2,2))(conv_1_pl)
    
    conv_2 = Conv2D(36, (5, 5), strides=(2,2), padding='same', name='conv_2')(conv_1_elu)
    conv_2_bn = BatchNormalization()(conv_2)
    conv_2_elu = ELU()(conv_2_bn)
    
    conv_2_pl = MaxPooling2D(pool_size=(2,2))(conv_2_elu)
    
    conc_1 = Concatenate(axis=3)([conv_1_pl, conv_2_elu])
    
    conv_3 = Conv2D(48, (5, 5), strides=(2,2), padding='same', name='conv_3')(conc_1)
    conv_3_bn = BatchNormalization()(conv_3)
    conv_3_elu = ELU()(conv_3_bn)
    
    conc_2 = Concatenate(axis=3)([conv_1_pl2, conv_3_elu, conv_2_pl])
    
    conv_4 = Conv2D(64, (3, 3), activation='elu', name='conv_4')(conc_2)
    conv_5 = Conv2D(64, (3, 3), activation='elu', name='conv2d_last')(conv_4)
    
    flat_1  = Flatten()(conv_5)
    fc_1 = Dense(100, activation='elu', name='fc_1')(flat_1)
    fc_2 = Dense(50,  activation='elu', name='fc_2')(fc_1)
    fc_3 = Dense(10,   activation='elu', name='fc_3')(fc_2)
    fc_last = Dense(1, name='fc_str')(fc_3)
    
    model = Model(inputs=img_input, output=fc_last)

    return model

def model_sap():
    img_shape = (config['input_image_height'],
                    config['input_image_width'],
                    config['input_image_depth'],)
    
    ######img model#######
    img_input = Input(shape=img_shape)
    lamb = Lambda(lambda x: x/127.5 - 1.0)(img_input)
    conv_1 = Conv2D(3, (1, 1), activation='elu')(lamb)
    conv_2 = Conv2D(32, (3, 3), activation='elu', name='conv2d_2')(conv_1)
    conv_3 = Conv2D(32, (3, 3), activation='elu', name='conv2d_3')(conv_2)
    pool_1 = MaxPooling2D(pool_size=(2, 2), name='maxpool_1')(conv_3)
    conv_4 = Conv2D(64, (3, 3), activation='elu', name='conv2d_5')(pool_1)
    conv_5 = Conv2D(64, (3, 3), activation='elu', name='conv2d_6')(conv_4)
    pool_2 = MaxPooling2D(pool_size=(2, 2), name='maxpool_2')(conv_5)
    conv_6 = Conv2D(128, (3, 3), activation='elu', name='conv2d_8')(pool_2)
    conv_7 = Conv2D(128, (3, 3), activation='elu', name='conv2d_last')(conv_6)
    pool_3 = MaxPooling2D(pool_size=(2, 2), name='maxpool_3')(conv_7)
    flat_1 = Flatten()(pool_3)
    fc_1 = Dense(512, activation='elu', name='fc_1')(flat_1)
    fc_2 = Dense(64,  activation='elu', name='fc_2')(fc_1)
    fc_3 = Dense(16,  activation='elu', name='fc_3')(fc_2)
    fc_last = Dense(1, name='fc_str')(fc_3)
    
    model = Model(inputs=img_input, output=fc_last)

    return model

def model_dave2sky():
    img_shape = (config['input_image_height'],
                    config['input_image_width'],
                    config['input_image_depth'],)
    
    ######img model#######
    img_input = Input(shape=img_shape)
    lamb = Lambda(lambda x: x/127.5 - 1.0)(img_input)
    conv_1 = Conv2D(24, (5, 5), strides=(2,2), activation='relu')(lamb)
    conv_1_bn = BatchNormalization()(conv_1)
    conv_2 = Conv2D(36, (5, 5), strides=(2,2), activation='relu')(conv_1_bn)
    conv_2_bn = BatchNormalization()(conv_2)
    conv_3 = Conv2D(48, (5, 5), strides=(2,2), activation='relu')(conv_2_bn)
    conv_3_bn = BatchNormalization()(conv_3)
    conv_4 = Conv2D(64, (3, 3), activation='relu')(conv_3_bn)
    conv_5 = Conv2D(64, (3, 3), name='conv2d_last', activation='relu')(conv_4)
    flat = Flatten()(conv_5)
    fc_1 = Dense(32, name='fc_1')(flat)
    fc_last = Dense(config['num_outputs'], name='fc_str')(fc_1)
    
    model = Model(inputs=img_input, output=fc_last)

    return model

def model_vgg16():    
    img_shape = (config['input_image_height'],
                    config['input_image_width'],
                    config['input_image_depth'],)
        
    img_input = Input(shape=img_shape)
    lamb = Lambda(lambda x: x/127.5 - 1.0)(img_input)
    conv_1 = Conv2D(64, (3, 3), padding="same", activation='relu', name='conv_1')(lamb)
    conv_2 = Conv2D(64, (3, 3), padding="same", activation='relu', name='conv_2')(conv_1)
    pool_1 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='maxpool_1')(conv_2)
    conv_3 = Conv2D(128, (3, 3), padding="same", activation='relu', name='conv_3')(pool_1)
    conv_4 = Conv2D(128, (3, 3), padding="same", activation='relu', name='conv_4')(conv_3)
    pool_2 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='maxpool_2')(conv_4)
    conv_5 = Conv2D(256, (3, 3), padding="same", activation='relu', name='conv_5')(pool_2)
    conv_6 = Conv2D(256, (3, 3), padding="same", activation='relu', name='conv_6')(conv_5)
    conv_7 = Conv2D(256, (3, 3), padding="same", activation='relu', name='conv_7')(conv_6)
    pool_3 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='maxpool_3')(conv_7)
    conv_8 = Conv2D(512, (3, 3), padding="same", activation='relu', name='conv_8')(pool_3)
    conv_9 = Conv2D(512, (3, 3), padding="same", activation='relu', name='conv_9')(conv_8)
    conv_10= Conv2D(512, (3, 3), padding="same", activation='relu', name='conv_10')(conv_9)
    pool_4 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='maxpool_4')(conv_10)
    conv_11= Conv2D(512, (3, 3), padding="same", activation='relu', name='conv_11')(pool_4)
    conv_12= Conv2D(512, (3, 3), padding="same", activation='relu', name='conv_12')(conv_11)
    conv_13= Conv2D(512, (3, 3), padding="same", activation='relu', name='conv_13')(conv_12)
    pool_5 = MaxPooling2D(pool_size=(2, 2), strides=(2, 2), name='maxpool_5')(conv_13)
    flat = Flatten()(pool_5)
    fc_1 = Dense(4096, activation='relu', name='fc_1')(flat)
    fc_2 = Dense(4096, activation='relu', name='fc_2')(fc_1)
    fc_last = Dense(1, activation='linear', name='fc_str')(fc_2)
    
    model = Model(inputs=img_input, output=fc_last)
    
    return model

def model_alexnet(): 
    img_shape = (config['input_image_height'],
                    config['input_image_width'],
                    config['input_image_depth'],)
        
    img_input = Input(shape=img_shape)
    lamb = Lambda(lambda x: x/127.5 - 1.0)(img_input)
    conv_1 = Conv2D(96, (11, 11), strides=(4,4), padding="same", activation='relu', name='conv_1')(lamb)
    conv_1_bn = BatchNormalization()(conv_1)
    conv_1_pl = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='maxpool_1')(conv_1_bn)
    
    conv_2 = Conv2D(256, (5, 5), padding="same", activation='relu', name='conv_2')(conv_1_pl)
    conv_2_bn = BatchNormalization()(conv_2)
    conv_2_pl = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='maxpool_2')(conv_2_bn)
    
    conv_3 = Conv2D(384, (3, 3), padding="same", activation='relu', name='conv_3')(conv_2_pl)
    conv_4 = Conv2D(384, (3, 3), padding="same", activation='relu', name='conv_4')(conv_3)
    conv_5 = Conv2D(256, (3, 3), padding="same", activation='relu', name='conv_5')(conv_4)
    conv_5_pl = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), name='maxpool_3')(conv_5)
    
    flat = Flatten()(conv_5_pl)
    fc_1 = Dense(4096, activation='relu', name='fc_1')(flat)
    fc_2 = Dense(4096, activation='relu', name='fc_2')(fc_1)
    fc_last = Dense(config['num_outputs'], activation='linear', name='fc_str')(fc_2)
    
    model = Model(inputs=img_input, output=fc_last)
    
    return model
    

def model_spatiotemporallstm():
    from keras.layers import ConvLSTM2D, Conv3D, LeakyReLU
    from keras.layers.wrappers import TimeDistributed
    
    img_shape = (3, config['input_image_height'],
                    config['input_image_width'],
                    config['input_image_depth'])
    
    input_img  = Input(shape=img_shape, name='input_image')
    lamb       = Lambda(lambda x: x/127.5 - 1.0, name='lamb_img')(input_img)
    convlstm_1 = ConvLSTM2D(64, 3, activation='relu', border_mode='same', return_sequences=True, name='convlstm_1')(lamb)
    batch_1    = BatchNormalization()(convlstm_1)
    convlstm_2 = ConvLSTM2D(32, 3, activation='relu', border_mode='same',return_sequences=True, name='convlstm_2')(batch_1)
    batch_2    = BatchNormalization()(convlstm_2)
    convlstm_3 = ConvLSTM2D(16, 3, activation='relu', border_mode='same',return_sequences=True, name='convlstm_3')(batch_2)
    batch_3    = BatchNormalization()(convlstm_3)
    convlstm_4 = ConvLSTM2D(8, 3, activation='relu', border_mode='same',return_sequences=True, name='convlstm_4')(batch_3)
    batch_4    = BatchNormalization()(convlstm_4)
    conv3d_1   = Conv3D(3, 3, activation='relu', border_mode='same', name='conv3d_last')(batch_4)
    # batch_5    = BatchNormalization()(conv3d_1)
    flat       = Flatten(name='flat')(conv3d_1)
    fc_1       = Dense(512,name='fc_1')(flat)
    leaky      = LeakyReLU(alpha=0.2)(fc_1)
    fc_last    = Dense(config['num_outputs'], activation='linear', name='fc_last')(leaky)

    model = Model(inputs=input_img, outputs=fc_last)
        
    return model

def model_lrcn():
    from keras.layers.recurrent import LSTM
    from keras.layers.wrappers import TimeDistributed

    # redefine input_shape to add one more dims
    img_shape = (None, config['input_image_height'],
                    config['input_image_width'],
                    config['input_image_depth'])
    
    input_img = Input(shape=img_shape, name='input_image')
    lamb      = TimeDistributed(Lambda(lambda x: x/127.5 - 1.0), name='lamb_img')(input_img)
    conv_1    = TimeDistributed(Convolution2D(24, (5, 5), activation='elu', strides=(2,2)), name='conv_1')(lamb)
    conv_2    = TimeDistributed(Convolution2D(36, (5, 5), activation='elu', strides=(2,2)), name='conv_2')(conv_1)
    conv_3    = TimeDistributed(Convolution2D(48, (5, 5), activation='elu', strides=(2,2)), name='conv_3')(conv_2)
    conv_4    = TimeDistributed(Convolution2D(64, (3, 3), activation='elu'), name='conv_4')(conv_3)
    conv_5    = TimeDistributed(Convolution2D(64, (3, 3), activation='elu'), name='conv2d_last')(conv_4)
    flat      = TimeDistributed(Flatten(), name='flat')(conv_5)
    lstm      = LSTM(  10, return_sequences=False, dropout=0.2, name='lstm')(flat)
    fc_1      = Dense(100, activation='elu', name='fc_1')(lstm)
    drop_1    = Dropout(0.2, name='drop_1')(fc_1)
    fc_2      = Dense( 50, activation='elu', name='fc_2')(drop_1)
    drop_2    = Dropout(0.2, name='drop_2')(fc_2)
    fc_3      = Dense( 10, activation='elu', name='fc_3')(drop_2)
    fc_last   = Dense(config['num_outputs'], activation='linear', name='fc_last')(fc_3)

    model = Model(inputs=input_img, outputs=fc_last)
    
    return model

def model_lrcn2():
    from keras.layers.recurrent import LSTM
    from keras.layers.wrappers import TimeDistributed

    # redefine input_shape to add one more dims
    img_shape = (None, config['input_image_height'],
                    config['input_image_width'],
                    config['input_image_depth'])
    
    input_img = Input(shape=img_shape, name='input_image')
    lamb      = TimeDistributed(Lambda(lambda x: x/127.5 - 1.0), name='lamb_img')(input_img)
    conv_1    = TimeDistributed(Convolution2D(24, (5, 5), activation='elu', strides=(2,2)), name='conv_1')(lamb)
    conv_2    = TimeDistributed(Convolution2D(36, (5, 5), activation='elu', strides=(2,2)), name='conv_2')(conv_1)
    conv_3    = TimeDistributed(Convolution2D(48, (5, 5), activation='elu', strides=(2,2)), name='conv_3')(conv_2)
    conv_4    = TimeDistributed(Convolution2D(64, (3, 3), activation='elu'), name='conv_4')(conv_3)
    conv_5    = TimeDistributed(Convolution2D(64, (3, 3), activation='elu'), name='conv2d_last')(conv_4)
    flat      = TimeDistributed(Flatten(), name='flat')(conv_5)
    lstm      = LSTM(  10, return_sequences=False, dropout=0.2, name='lstm')(flat)
    fc_1      = Dense(100, activation='elu', name='fc_1')(lstm)
    drop_1    = Dropout(0.2, name='drop_1')(fc_1)
    fc_2      = Dense( 50, activation='elu', name='fc_2')(drop_1)
    drop_2    = Dropout(0.2, name='drop_2')(fc_2)
    fc_3      = Dense( 10, activation='elu', name='fc_3')(drop_2)
    fc_last   = Dense(config['num_outputs'], activation='linear', name='fc_last')(fc_3)

    model = Model(inputs=input_img, outputs=fc_last)
    
    return model
    
def model_lrcn3():
    from keras.layers.recurrent import LSTM
    from keras.layers.wrappers import TimeDistributed

    # redefine input_shape to add one more dims
    img_shape = (None, config['input_image_height'],
                    config['input_image_width'],
                    config['input_image_depth'])
    
    input_img = Input(shape=img_shape, name='input_image')
    lamb      = TimeDistributed(Lambda(lambda x: x/127.5 - 1.0), name='lamb_img')(input_img)
    conv_1    = TimeDistributed(Convolution2D(24, (5, 5), activation='elu', strides=(2,2)), name='conv_1')(lamb)
    conv_2    = TimeDistributed(Convolution2D(36, (5, 5), activation='elu', strides=(2,2)), name='conv_2')(conv_1)
    conv_3    = TimeDistributed(Convolution2D(48, (5, 5), activation='elu', strides=(2,2)), name='conv_3')(conv_2)
    conv_4    = TimeDistributed(Convolution2D(64, (3, 3), activation='elu'), name='conv_4')(conv_3)
    conv_5    = TimeDistributed(Convolution2D(64, (3, 3), activation='elu'), name='conv2d_last')(conv_4)
    flat      = TimeDistributed(Flatten(), name='flat')(conv_5)
    lstm1      = LSTM(10, return_sequences=True, name='lstm_1')(flat)
    lstm2      = LSTM(10, return_sequences=True, name='lstm_2')(lstm1)
    lstm3      = LSTM(10, return_sequences=False, name='lstm_3')(lstm2)
    fc_4      = Dense(  10, activation='elu', name='fc_4')(lstm3)
    fc_last   = Dense(config['num_outputs'], activation='linear', name='fc_last')(fc_4)

    model = Model(inputs=input_img, outputs=fc_last)
    
    return model

def model_cooplrcn():
    from keras.layers.recurrent import LSTM
    from keras.layers.wrappers import TimeDistributed

    # redefine input_shape to add one more dims
    img_shape = (None, config['input_image_height'],
                    config['input_image_width'],
                    config['input_image_depth'])
    
    input_img = Input(shape=img_shape, name='input_image')
    lamb      = TimeDistributed(Lambda(lambda x: x/127.5 - 1.0), name='lamb_img')(input_img)
    conv_1    = TimeDistributed(Convolution2D(24, (5, 5), activation='relu', strides=(5,4)), name='conv_1')(lamb)
    print(conv_1.shape)
    conv_2    = TimeDistributed(Convolution2D(32, (5, 5), activation='relu', strides=(3,2)), name='conv_2')(conv_1)
    print(conv_2.shape)
    conv_3    = TimeDistributed(Convolution2D(48, (5, 5), activation='relu', strides=(5,4)), name='conv_3')(conv_2)
    print(conv_3.shape)
    conv_4    = TimeDistributed(Convolution2D(64, (5, 5), activation='relu', strides=(1,1)), name='conv_4')(conv_3)
    print(conv_4.shape)
    conv_5    = TimeDistributed(Convolution2D(128, (5, 5), activation='relu', strides=(1,2)), name='conv2d_last')(conv_4)
    print(conv_5.shape)
    flat      = TimeDistributed(Flatten(), name='flat')(conv_5)
    lstm_1    = LSTM(64, return_sequences=True, name='lstm_1')(flat)
    lstm_2    = LSTM(64, return_sequences=True, name='lstm_2')(lstm_1)
    lstm_3    = LSTM(64, return_sequences=False, name='lstm_3')(lstm_2)
    fc_1      = Dense(100, activation='relu', name='fc_1')(lstm_3)
    fc_2      = Dense(50, activation='relu' , name='fc_2')(fc_1)
    fc_3      = Dense(10, activation='relu', name='fc_3')(fc_2)
    fc_last   = Dense(config['num_outputs'], activation='linear', name='fc_last')(fc_3)

    model = Model(inputs=input_img, outputs=fc_last)
    
    return model

class NetModel:
    def __init__(self, model_path):
        self.model = None
        model_name = model_path[model_path.rfind('/'):] # get folder name
        self.name = model_name.strip('/')

        self.model_path = model_path
        #self.config = Config()

        # to address the error:
        #   Could not create cudnn handle: CUDNN_STATUS_INTERNAL_ERROR
        os.environ["CUDA_VISIBLE_DEVICES"]=str(config['gpus'])
        
        gpu_options = tf.GPUOptions(allow_growth=True)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        K.tensorflow_backend.set_session(sess)
        # with tf.device('/cpu:0'):
        self._model()

    ###########################################################################
    #
    def _model(self):
        if config['network_type'] == const.NET_TYPE_JAEROCK:
            self.model = model_jaerock()
        elif config['network_type'] == const.NET_TYPE_JAEROCK_ELU:
            self.model = model_jaerock_elu()
        elif config['network_type'] == const.NET_TYPE_CE491:
            self.model = model_ce491()
        elif config['network_type'] == const.NET_TYPE_JAEROCK_VEL:
            self.model = model_jaerock_vel()
        elif config['network_type'] == const.NET_TYPE_JAEROCK_ELU_360:
            self.model = model_jaerock_elu()
        elif config['network_type'] == const.NET_TYPE_SAP:
            self.model = model_sap()
        elif config['network_type'] == const.NET_TYPE_DAVE2SKY:
            self.model = model_dave2sky()
        elif config['network_type'] == const.NET_TYPE_VGG16:
            self.model = model_vgg16()
        elif config['network_type'] == const.NET_TYPE_ALEXNET:
            self.model = model_alexnet()
        elif config['network_type'] == const.NET_TYPE_DONGHYUN:
            self.model = model_donghyun()
        elif config['network_type'] == const.NET_TYPE_DONGHYUN2:
            self.model = model_donghyun2()
        elif config['network_type'] == const.NET_TYPE_DONGHYUN3:
            self.model = model_donghyun3()
        elif config['network_type'] == const.NET_TYPE_DONGHYUN4:
            self.model = model_donghyun4()
        elif config['network_type'] == const.NET_TYPE_DONGHYUN5:
            self.model = model_donghyun5()
        elif config['network_type'] == const.NET_TYPE_DONGHYUN6:
            self.model = model_donghyun6()
        elif config['network_type'] == const.NET_TYPE_DONGHYUN7:
            self.model = model_donghyun7()
        elif config['network_type'] == const.NET_TYPE_DONGHYUN8:
            self.model = model_donghyun8()
            
        elif config['network_type'] == const.NET_TYPE_LRCN:
            self.model = model_lrcn()
        elif config['network_type'] == const.NET_TYPE_LRCN2:
            self.model = model_lrcn2()
        elif config['network_type'] == const.NET_TYPE_LRCN3:
            self.model = model_lrcn3()
        elif config['network_type'] == const.NET_TYPE_SPTEMLSTM:
            self.model = model_spatiotemporallstm()
        elif config['network_type'] == const.NET_TYPE_COOPLRCN:
            self.model = model_cooplrcn()
        else:
            exit('ERROR: Invalid neural network type.')

        self.summary()
        self._compile()



    # ###########################################################################
    # #
    # def _mean_squared_error(self, y_true, y_pred):
    #     diff = K.abs(y_true - y_pred)
    #     if (diff < config['steering_angle_tolerance']) is True:
    #         diff = 0
    #     return K.mean(K.square(diff))

    ###########################################################################
    #
    def _compile(self):
        if config['lstm'] is True:
            learning_rate = config['lstm_lr']
        else:
            learning_rate = config['cnn_lr']
        decay = config['decay']
        self.model.compile(loss=losses.mean_squared_error,
                    optimizer=optimizers.Adam(lr=learning_rate, decay=decay), 
                    metrics=['accuracy'])
        # if config['steering_angle_tolerance'] == 0.0:
        #     self.model.compile(loss=losses.mean_squared_error,
        #               optimizer=optimizers.Adam(),
        #               metrics=['accuracy'])
        # else:
        #     self.model.compile(loss=losses.mean_squared_error,
        #               optimizer=optimizers.Adam(),
        #               metrics=['accuracy', self._mean_squared_error])


    ###########################################################################
    #
    # save model
    def save(self, model_name):

        json_string = self.model.to_json()
        #weight_filename = self.model_path + '_' + Config.config_yaml_name \
        #    + '_N' + str(config['network_type'])
        open(model_name+'.json', 'w').write(json_string)
        self.model.save_weights(model_name+'.h5', overwrite=True)


    ###########################################################################
    # model_path = '../data/2007-09-22-12-12-12.
    def weight_load(self, load_model_name):

        from keras.models import model_from_json

        json_string = self.model.to_json()
        open(load_model_name+'.json', 'w').write(json_string)
        self.model = model_from_json(open(load_model_name+'.json').read())
        self.model.load_weights(load_model_name)
        self._compile()
    
    
    def load(self):

        from keras.models import model_from_json

        self.model = model_from_json(open(self.model_path+'.json').read())
        self.model.load_weights(self.model_path+'.h5')
        self._compile()

    ###########################################################################
    #
    # show summary
    def summary(self):
        self.model.summary()

