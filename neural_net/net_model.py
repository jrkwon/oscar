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

# def model_convlstm():
#     from keras.layers.recurrent import LSTM
#     from keras.layers.wrappers import TimeDistributed

#     # redefine input_shape to add one more dims
#     input_shape = (None, config['input_image_height'],
#                             config['input_image_width'],
#                             config['input_image_depth'])
#     return Sequential([
#         TimeDistributed(Lambda(lambda x: x/127.5 - 1.0), input_shape=input_shape),
#         TimeDistributed(Conv2D(24, (5, 5), strides=(2,2))),
#         TimeDistributed(Conv2D(36, (5, 5), strides=(2,2))),
#         TimeDistributed(Conv2D(48, (5, 5), strides=(2,2))),
#         TimeDistributed(Conv2D(64, (3, 3))),
#         TimeDistributed(Conv2D(64, (3, 3), name='conv2d_last')),
#         TimeDistributed(Flatten()),
#         Dense(1000),
#         LSTM(return_sequences=False, units=10),
#         Dense(100),
#         Dense(50),
#         Dense(10),
#         Dense(config['num_outputs'])])

def model_jaerock_lstm_vel():
    from keras.layers.recurrent import LSTM
    from keras.layers.wrappers import TimeDistributed

    # redefine input_shape to add one more dims
    img_shape = (None, config['input_image_height'],
                    config['input_image_width'],
                    config['input_image_depth'])
    
    ######img model#######
    input_img = Input(shape=img_shape, name='input_image')
    lamb_img  = TimeDistributed(Lambda(lambda x: x/127.5 - 1.0), name='lamb')(input_img)
    conv_1    = TimeDistributed(Convolution2D(24, (5, 5), strides=(2,2)), name='conv_1')(lamb_img)
    conv_2    = TimeDistributed(Convolution2D(36, (5, 5), strides=(2,2)), name='conv_2')(conv_1)
    conv_3    = TimeDistributed(Convolution2D(48, (5, 5), strides=(2,2)), name='conv_3')(conv_2)
    conv_4    = TimeDistributed(Convolution2D(64, (3, 3)), name='conv_4')(conv_3)
    conv_5    = TimeDistributed(Convolution2D(64, (3, 3)), name='conv2d_last')(conv_4)
    flat      = TimeDistributed(Flatten(), name='flat')(conv_5)
    fc_1      = TimeDistributed(Dense(1000, activation='relu'), name='fc_1')(flat)
    fc_2      = TimeDistributed(Dense(100, activation='relu' ), name='fc_2')(fc_1)
    
    ##########velocity###############
    # input_velocity = Input(shape=(None,  config['input_velocity']), name='input_velocity')
    # # lamb_vel  = TimeDistributed(Lambda(lambda x: x/(config['max_vel']/2.0) - 1.0), name='lamb')(input_velocity)
    # fc_vel_1  = TimeDistributed(Dense(50, activation='relu'), name='fc_vel')(input_velocity)
    # #################################
    # ##########concat#################
    # concat    = concatenate([fc_2, fc_vel_1], name='concat')
    lstm      = LSTM(10, return_sequences=False, name='lstm')(fc_2)
    fc_3      = TimeDistributed(Dense(50, activation='relu'), name='fc_3')(lstm)
    fc_4      = TimeDistributed(Dense(10, activation='relu'), name='fc_4')(fc_3)
    fc_last   = TimeDistributed(Dense(config['num_outputs'], activation='linear'), name='fc_str')(fc_4)
    # print(fc_last.shape)
    model = Model(inputs=[input_img], outputs=fc_last)
    # model = Model(inputs=[input_img, input_velocity], outputs=fc_last)
    
    return model

def model_convlstm():
    from keras.layers.recurrent import LSTM
    from keras.layers.wrappers import TimeDistributed

    # redefine input_shape to add one more dims
    img_shape = (None, config['input_image_height'],
                    config['input_image_width'],
                    config['input_image_depth'])
    
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
    
    lstm      = LSTM(10, return_sequences=True, name='lstm')(fc_2)
    fc_3      = TimeDistributed(Dense(50, activation='relu'), name='fc_3')(lstm)
    fc_4      = TimeDistributed(Dense(10, activation='relu'), name='fc_4')(fc_3)
    fc_last   = TimeDistributed(Dense(1, activation='linear'), name='fc_last')(fc_4)

    model = Model(inputs=input_img, outputs=fc_last)
        
    
    return model

class NetModel:
    def __init__(self, model_path):
        self.model = None
        self.multi_gpu_model = None
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
        from keras.utils import multi_gpu_model
        
        if config['network_type'] == const.NET_TYPE_JAEROCK:
            self.model = model_jaerock()
        elif config['network_type'] == const.NET_TYPE_CE491:
            self.model = model_ce491()
        elif config['network_type'] == const.NET_TYPE_CONVLSTM:
            self.model = model_convlstm()
        elif config['network_type'] == const.NET_TYPE_JAEROCK_LSTM_VEL:
            self.model = model_jaerock_lstm_vel()
        else:
            print('Neural network type is not defined.')
            return
        
        if config['num_gpus'] >= 2:
            self.multi_gpu_model = multi_gpu_model(self.model, gpus=config['num_gpus'])
        
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
        if config['lstm'] is True:
            learning_rate = config['lstm_lr']
        else:
            learning_rate = config['cnn_lr']

        if config['num_gpus'] >= 2:
            self.multi_gpu_model.compile(loss=losses.mean_squared_error,
                        optimizer=optimizers.Adam(lr=learning_rate), 
                        metrics=['accuracy'])
        else:
            self.model.compile(loss=losses.mean_squared_error,
                        optimizer=optimizers.Adam(lr=learning_rate), 
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

