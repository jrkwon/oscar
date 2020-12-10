#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 13:23:14 2017
History:
11/28/2020: modified for OSCAR 

@author: jaerock
"""

import matplotlib.pyplot as plt
import cv2
import numpy as np
#import keras
import sklearn

import const
from net_model import NetModel
from drive_data import DriveData
from config import Config
from image_process import ImageProcess
from data_augmentation import DataAugmentation

config = Config.config

###############################################################################
#
class DriveTrain:
    
    ###########################################################################
    # data_path = 'path_to_drive_data'  e.g. ../data/2017-09-22-10-12-34-56/'
    def __init__(self, data_path):
        
        if data_path[-1] == '/':
            data_path = data_path[:-1]

        loc_slash = data_path.rfind('/')
        if loc_slash != -1: # there is '/' in the data path
            model_name = data_path[loc_slash + 1:] # get folder name
            #model_name = model_name.strip('/')
        else:
            model_name = data_path
        csv_path = data_path + '/' + model_name + const.DATA_EXT  # use it for csv file name 
        
        self.csv_path = csv_path
        self.train_generator = None
        self.valid_generator = None
        self.train_hist = None
        self.drive = None
        
        #self.config = Config() #model_name)
        
        self.data_path = data_path
        #self.model_name = model_name

        self.model_name = data_path + '_' + Config.config_yaml_name \
            + '_N' + str(config['network_type'])
        self.model_ckpt_name = self.model_name + '_ckpt'

        
        self.drive = DriveData(self.csv_path)
        self.net_model = NetModel(data_path)
        self.image_process = ImageProcess()
        self.data_aug = DataAugmentation()
        
        
    ###########################################################################
    #
    def _prepare_data(self, normalize_data):
    
        self.drive.read(normalize_data = normalize_data)
        
        from sklearn.model_selection import train_test_split
        
        samples = list(zip(self.drive.image_names, self.drive.measurements))
        self.train_data, self.valid_data = train_test_split(samples, 
                                   test_size=config['validation_rate'])
        
        self.num_train_samples = len(self.train_data)
        self.num_valid_samples = len(self.valid_data)
        
        print('Train samples: ', self.num_train_samples)
        print('Valid samples: ', self.num_valid_samples)
    
                                          
    ###########################################################################
    #
    def _build_model(self, show_summary=True):

        def _generator(samples, batch_size=config['batch_size']):
            num_samples = len(samples)
            while True: # Loop forever so the generator never terminates
               
                if config['lstm'] is False:
                    samples = sklearn.utils.shuffle(samples)

                for offset in range(0, num_samples, batch_size):
                    batch_samples = samples[offset:offset+batch_size]
        
                    images = []
                    measurements = []

                    for image_name, measurement in batch_samples:
                        
                        image_path = self.data_path + '/' + image_name
                        image = cv2.imread(image_path)

                        # if collected data is not cropped then crop here
                        # otherwise do not crop.
                        if config['crop'] is not True:
                            image = image[config['image_crop_y1']:config['image_crop_y2'],
                                          config['image_crop_x1']:config['image_crop_x2']]

                        image = cv2.resize(image, 
                                           (config['input_image_width'],
                                            config['input_image_height']))
                        image = self.image_process.process(image)
                        images.append(image)
        
                        steering_angle, throttle = measurement
                        
                        if abs(steering_angle) < config['steering_angle_jitter_tolerance']:
                            steering_angle = 0
                        
                        measurements.append(steering_angle*config['steering_angle_scale'])
                        
                        if config['data_aug_flip'] is True:    
                            # Flipping the image
                            flip_image, flip_steering = self.data_aug.flipping(image, steering_angle)
                            images.append(flip_image)
                            measurements.append(flip_steering*config['steering_angle_scale'])
    
                        if config['data_aug_bright'] is True:    
                            # Changing the brightness of image
                            if steering_angle > config['steering_angle_jitter_tolerance'] or \
                                steering_angle < -config['steering_angle_jitter_tolerance']:
                                bright_image = self.data_aug.brightness(image)
                                images.append(bright_image)
                                measurements.append(steering_angle*config['steering_angle_scale'])
    
                        if config['data_aug_shift'] is True:    
                            # Shifting the image
                            shift_image, shift_steering = self.data_aug.shift(image, steering_angle)
                            images.append(shift_image)
                            measurements.append(shift_steering*config['steering_angle_scale'])

                    X_train = np.array(images)
                    y_train = np.array(measurements)

                    if config['lstm'] is True:
                        X_train = np.array(images).reshape(-1, 1, 
                                          config['input_image_height'],
                                          config['input_image_width'],
                                          config['input_image_depth'])
                        y_train = np.array(measurements).reshape(-1, 1, 1)
                    
                    if config['lstm'] is False:
                        yield sklearn.utils.shuffle(X_train, y_train)
                    else:
                        yield X_train, y_train
        
        self.train_generator = _generator(self.train_data)
        self.valid_generator = _generator(self.valid_data)
        
        if (show_summary):
            self.net_model.model.summary()
    
    ###########################################################################
    #
    def _start_training(self):
        
        if (self.train_generator == None):
            raise NameError('Generators are not ready.')
        
        ######################################################################
        # callbacks
        from keras.callbacks import ModelCheckpoint, EarlyStopping, TensorBoard
        
        # checkpoint
        callbacks = []
        #weight_filename = self.data_path + '_' + Config.config_yaml_name \
        #    + '_N' + str(config['network_type']) + '_ckpt'
        checkpoint = ModelCheckpoint(self.model_ckpt_name +'.h5',
                                     monitor='val_loss', 
                                     verbose=1, save_best_only=True, mode='min')
        callbacks.append(checkpoint)
        
        # early stopping
        patience = config['early_stopping_patience']
        earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=patience, 
                                  verbose=1, mode='min')
        callbacks.append(earlystop)

        # tensor board
        tensorboard = TensorBoard(log_dir='./logs')
        callbacks.append(tensorboard)


        self.train_hist = self.net_model.model.fit_generator(
                self.train_generator, 
                steps_per_epoch=self.num_train_samples//config['batch_size'], 
                epochs=config['num_epochs'], 
                validation_data=self.valid_generator,
                validation_steps=self.num_valid_samples//config['batch_size'],
                verbose=1, callbacks=callbacks, 
                use_multiprocessing=True)
    

    ###########################################################################
    #
    def _plot_training_history(self):
    
        print(self.train_hist.history.keys())
        
        plt.figure() # new figure window
        ### plot the training and validation loss for each epoch
        plt.plot(self.train_hist.history['loss'][1:])
        plt.plot(self.train_hist.history['val_loss'][1:])
        plt.title('model mean squared error loss')
        plt.ylabel('mse loss')
        plt.xlabel('epoch')
        plt.legend(['training set', 'validatation set'], loc='upper right')
        #plt.show()
        plt.savefig(self.model_name + '_model.png')
        
        
    ###########################################################################
    #
    def train(self, show_summary=True, normalize_data=True):
        
        self._prepare_data(normalize_data)
        self._build_model(show_summary)
        self._start_training()
        self.net_model.save(self.model_name)
        self._plot_training_history()
        Config.summary()
