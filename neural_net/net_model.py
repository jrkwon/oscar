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
from keras.layers import Lambda, Dropout, Flatten, Dense, Activation, Concatenate
from keras.layers import Conv2D, MaxPooling2D, BatchNormalization, Input
from keras import losses, optimizers
import keras.backend as K
import tensorflow as tf

import const
from config import Config

config = Config.neural_net

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
        Dense(config['num_outputs'])])    

def model_convlstm():
    from keras.layers.recurrent import LSTM
    from keras.layers.wrappers import TimeDistributed

    # redefine input_shape to add one more dims
    input_shape = (None, config['input_image_height'],
                            config['input_image_width'],
                            config['input_image_depth'])
    return Sequential([
        TimeDistributed(Lambda(lambda x: x/127.5 - 1.0), input_shape=input_shape),
        TimeDistributed(Conv2D(24, (5, 5), strides=(2,2))),
        TimeDistributed(Conv2D(36, (5, 5), strides=(2,2))),
        TimeDistributed(Conv2D(48, (5, 5), strides=(2,2))),
        TimeDistributed(Conv2D(64, (3, 3))),
        TimeDistributed(Conv2D(64, (3, 3), name='conv2d_last')),
        TimeDistributed(Flatten()),
        LSTM(return_sequences=False, units=10),
        Dense(1000),
        Dense(100),
        Dense(50),
        Dense(10),
        Dense(config['num_outputs'])])

class NetModel:
    def __init__(self, model_path):
        self.model = None
        model_name = model_path[model_path.rfind('/'):] # get folder name
        self.name = model_name.strip('/')

        self.model_path = model_path
        #self.config = Config()

        # to address the error:
        #   Could not create cudnn handle: CUDNN_STATUS_INTERNAL_ERROR
        gpu_options = tf.GPUOptions(allow_growth=True)
        sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
        K.tensorflow_backend.set_session(sess)

        self._model()

    ###########################################################################
    #
    def _model(self):
        if config['network_type'] == const.NET_TYPE_CE491:
            self.model = model_ce491()
        elif config['network_type'] == const.NET_TYPE_JAEROCK:
            self.model = model_jaerock()
        elif config['network_type'] == const.NET_TYPE_CONVLSTM:
            self.model = model_convlstm()
        else:
            print('Neural network type is not defined.')
            return

        self.model.summary()
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
        self.model.compile(loss=losses.mean_squared_error,
                    optimizer=optimizers.Adam(),
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

