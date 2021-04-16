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
        Conv2D(24, (5, 5), strides=(2,2), activation='relu'),
        Conv2D(36, (5, 5), strides=(2,2), activation='relu'),
        Conv2D(48, (5, 5), strides=(2,2), activation='relu'),
        Conv2D(64, (3, 3), activation='relu'),
        Conv2D(64, (3, 3), activation='relu', name='conv2d_last'),
        Flatten(),
        Dense(100, activation='relu'),
        Dense(50, activation='relu'),
        Dense(10, activation='relu'),
        Dense(config['num_outputs'])])

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
    conv_3 = Conv2D(36, (6, 6), strides=(2,2), activation='elu')(conv_2)
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

def model_donghyun4(): #필터사이즈를 donghyun3에 비해 전체적으로 더 크게
    img_shape = (config['input_image_height'],
                    config['input_image_width'],
                    config['input_image_depth'],)
    
    ######img model#######
    img_input = Input(shape=img_shape)
    lamb = Lambda(lambda x: x/127.5 - 1.0)(img_input)
    conv_1 = Conv2D(64, (8, 8), strides=(4,4), activation='elu')(lamb)
    conv_2 = Conv2D(48, (8, 8), strides=(2,2), activation='elu')(conv_1)
    conv_3 = Conv2D(36, (6, 6), strides=(2,2), activation='elu')(conv_2)
    conv_4 = Conv2D(24, (5, 5), activation='elu')(conv_3)
    conv_5 = Conv2D(24, (3, 3), activation='elu', name='conv2d_last')(conv_4)
    flat = Flatten()(conv_5)
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
    conv_1 = Conv2D(64, (8, 8), strides=(4,4), activation='elu')(lamb)
    conv_2 = Conv2D(48, (8, 8), strides=(2,2), activation='elu', name='conv2d_2')(conv_1)
    conv_3 = Conv2D(36, (6, 6), strides=(2,2), activation='elu', name='conv2d_3')(conv_2)
    conv_4 = Conv2D(24, (5, 5), activation='elu', name='conv2d_4')(conv_3)
    conv_5 = Conv2D(24, (3, 3), activation='elu', name='conv2d_last')(conv_4)
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


    
def model_spatiotemporallstm():
    from keras.layers import ConvLSTM2D, Convolution3D
    from keras.layers.wrappers import TimeDistributed
    
    img_shape = (None, config['input_image_height'],
                    config['input_image_width'],
                    config['input_image_depth'])
    
    leaky_relu = tf.nn.leaky_relu
    
    input_img  = Input(shape=img_shape, name='input_image')
    lamb       = Lambda(lambda x: x/127.5 - 1.0, name='lamb_img')(input_img)
    convlstm_1 = ConvLSTM2D(64, (3, 3), strides=(2,2), activation='relu', return_sequences=True, name='convlstm_1')(lamb)
    batch_1    = BatchNormalization()(convlstm_1)
    convlstm_2 = ConvLSTM2D(32, (3, 3), strides=(2,2), activation='relu', return_sequences=True, name='convlstm_2')(batch_1)
    batch_2    = BatchNormalization()(convlstm_2)
    convlstm_3 = ConvLSTM2D(16, (3, 3), strides=(2,2), activation='relu', return_sequences=True, name='convlstm_3')(batch_2)
    batch_3    = BatchNormalization()(convlstm_3)
    convlstm_4 = ConvLSTM2D(8, (3, 3), activation='relu', return_sequences=True, name='convlstm_4')(batch_3)
    batch_4    = BatchNormalization()(convlstm_4)
    conv3d_1   = Convolution3D(3, (3, 3, 3), activation='relu', name='conv2d_last')(batch_4)
    flat       = Flatten(name='flat')(conv3d_1)
    fc_1       = Dense(512, activation=leaky_relu, name='fc_1')(flat)
    fc_last    = Dense(config['num_outputs'], activation='linear', name='fc_last')(fc_1)

    model = Model(inputs=input_img, outputs=fc_last)
        
    return model

def model_lrcn():
    from keras.layers.recurrent import LSTM
    from keras.layers.wrappers import TimeDistributed

    # redefine input_shape to add one more dims
    img_shape = (None, config['input_image_height'],
                    config['input_image_width'],
                    config['input_image_depth'])
    vel_shape = (None, 1)
    
    input_img = Input(shape=img_shape, name='input_image')
    lamb      = TimeDistributed(Lambda(lambda x: x/127.5 - 1.0), name='lamb_img')(input_img)
    conv_1    = TimeDistributed(Convolution2D(24, (5, 5), strides=(2,2)), name='conv_1')(lamb)
    conv_2    = TimeDistributed(Convolution2D(36, (5, 5), strides=(2,2)), name='conv_2')(conv_1)
    conv_3    = TimeDistributed(Convolution2D(48, (5, 5), strides=(2,2)), name='conv_3')(conv_2)
    conv_4    = TimeDistributed(Convolution2D(64, (3, 3)), name='conv_4')(conv_3)
    conv_5    = TimeDistributed(Convolution2D(64, (3, 3)), name='conv2d_last')(conv_4)
    flat      = TimeDistributed(Flatten(), name='flat')(conv_5)
    fc_1      = TimeDistributed(Dense(1000, activation='relu'), name='fc_1')(flat)
    fc_2      = TimeDistributed(Dense(100, activation='relu' ), name='fc_2')(fc_1)
    
    if config['num_inputs'] == 1:
        lstm      = LSTM(10, return_sequences=False, name='lstm')(fc_2)
        fc_3      = Dense(50, activation='relu', name='fc_3')(lstm)
        fc_4      = Dense(10, activation='relu', name='fc_4')(fc_3)
        fc_last   = Dense(config['num_outputs'], activation='linear', name='fc_last')(fc_4)
    
        model = Model(inputs=input_img, outputs=fc_last)
        
    elif config['num_inputs'] == 2:
        input_velocity = Input(shape=vel_shape, name='input_velocity')
        lamb      = TimeDistributed(Lambda(lambda x: x / 38), name='lamb_vel')(input_velocity)
        fc_vel_1  = TimeDistributed(Dense(50, activation='relu'), name='fc_vel')(lamb)
        concat    = concatenate([fc_2, fc_vel_1], name='concat')
        lstm      = LSTM(3, return_sequences=False, name='lstm')(concat)
        fc_3      = Dense(50, activation='relu', name='fc_3')(lstm)
        fc_4      = Dense(10, activation='relu', name='fc_4')(fc_3)
        fc_last   = Dense(config['num_outputs'], activation='linear', name='fc_last')(fc_4)

        model = Model(inputs=[input_img, input_velocity], outputs=fc_last)
    
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
        self._model()

    ###########################################################################
    #
    def _model(self):
        # if config['network_type'] == const.NET_TYPE_JAEROCK:
        #     self.model = model_jaerock()
        if config['network_type'] == const.NET_TYPE_JAEROCK_ELU:
            self.model = model_jaerock_elu()
        elif config['network_type'] == const.NET_TYPE_CE491:
            self.model = model_ce491()
        elif config['network_type'] == const.NET_TYPE_JAEROCK_VEL:
            self.model = model_jaerock_vel()
        elif config['network_type'] == const.NET_TYPE_JAEROCK_ELU_850:
            self.model = model_jaerock_elu()
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
            
        elif config['network_type'] == const.NET_TYPE_LRCN:
            self.model = model_lrcn()
        elif config['network_type'] == const.NET_TYPE_SPTEMLSTM:
            self.model = model_spatiotemporallstm()
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

