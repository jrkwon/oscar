#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 13:23:14 2017
History:
11/28/2020: modified for OSCAR 

@author: jaerock
"""

from datetime import datetime
import matplotlib.pyplot as plt
import cv2
import numpy as np
#import keras
import sklearn
from sklearn.model_selection import train_test_split

import const
from net_model import NetModel
from drive_data import DriveData
from config import Config
from image_process import ImageProcess
from data_augmentation import DataAugmentation

config = Config.neural_net

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

        self.model_name = data_path + '_' + Config.neural_net_yaml_name \
            + '_N' + str(config['network_type'])
        self.model_ckpt_name = self.model_name + '_ckpt'

        
        self.drive = DriveData(self.csv_path)
        self.net_model = NetModel(data_path)
        self.image_process = ImageProcess()
        self.data_aug = DataAugmentation()
        
        
    ###########################################################################
    #
    def _prepare_data(self):
    
        self.drive.read()
        
        samples = list(zip(self.drive.image_names, self.drive.measurements, self.drive.velocities))

        if config['lstm'] is True:
            self.train_data, self.valid_data = self._prepare_lstm_data(samples)
        else:    
            self.train_data, self.valid_data = train_test_split(samples, 
                                       test_size=config['validation_rate'])
        
        self.num_train_samples = len(self.train_data)
        self.num_valid_samples = len(self.valid_data)
        
        print('Train samples: ', self.num_train_samples)
        print('Valid samples: ', self.num_valid_samples)
    
                                          
    ###########################################################################
    # group the samples by the number of timesteps
    def _prepare_lstm_data(self, samples):
        num_samples = len(samples)

        # get the last index number      
        steps = 1
        last_index = (num_samples - config['lstm_timestep'])//steps
        
        image_names = []
        measurements = []
        velocities = []
        for i in range(0, last_index, steps):
            sub_samples = samples[ i : i+config['lstm_timestep'] ]
            
            # print('num_batch_sample : ',len(batch_samples))
            sub_image_names = []
            sub_measurements = []
            sub_velocities = []
            for image_name, measurment, velocity in sub_samples:
                sub_image_names.append(image_name)
                sub_measurements.append(measurment)
                sub_velocities.append(velocity)

            image_names.append(sub_image_names)
            measurements.append(sub_measurements)
            velocities.append(sub_velocities)
        
        samples = list(zip(image_names, measurements, velocities))
        return train_test_split(samples, test_size=config['validation_rate'], 
                                shuffle=False)        


    ###########################################################################
    #
    def _build_model(self, show_summary=True):

        def _data_augmentation(image, steering_angle):
            if config['data_aug_flip'] is True:    
                # Flipping the image
                return True, self.data_aug.flipping(image, steering_angle)

            if config['data_aug_bright'] is True:    
                # Changing the brightness of image
                if steering_angle > config['steering_angle_jitter_tolerance'] or \
                    steering_angle < -config['steering_angle_jitter_tolerance']:
                    image = self.data_aug.brightness(image)
                return True, image, steering_angle

            if config['data_aug_shift'] is True:    
                # Shifting the image
                return True, self.data_aug.shift(image, steering_angle)

            return False, image, steering_angle

        def _prepare_batch_samples(batch_samples):
            images = []
            measurements = []

            for image_name, measurement in batch_samples:
                
                image_path = self.data_path + '/' + image_name
                image = cv2.imread(image_path)

                # if collected data is not cropped then crop here
                # otherwise do not crop.
                if Config.data_collection['crop'] is not True:
                    image = image[Config.data_collection['image_crop_y1']:Config.data_collection['image_crop_y2'],
                                  Config.data_collection['image_crop_x1']:Config.data_collection['image_crop_x2']]

                image = cv2.resize(image, 
                                    (config['input_image_width'],
                                    config['input_image_height']))
                image = self.image_process.process(image)
                images.append(image)

                steering_angle, throttle = measurement
                
                if abs(steering_angle) < config['steering_angle_jitter_tolerance']:
                    steering_angle = 0
                
                measurements.append(steering_angle*config['steering_angle_scale'])
                
                # data augmentation
                append, image, steering_angle = _data_augmentation(image, steering_angle)
                if append is True:
                    images.append(image)
                    measurements.append(steering_angle*config['steering_angle_scale'])

            return images, measurements

        def _prepare_lstm_batch_samples(batch_samples):
            images = []
            measurements = []
            velocities = []
            throttles = []
            for i in range(0, config['batch_size']):

                images_timestep = []
                #images_names_timestep = []
                measurements_timestep = []
                throttles_timestep = []
                velocities_timestep = []
                
                for j in range(0, config['lstm_timestep']):

                    image_name = batch_samples[i][0][j]
                    image_path = self.data_path + '/' + image_name
                    image = cv2.imread(image_path)

                    velocity = batch_samples[i][2][j]
                    # if collected data is not cropped then crop here
                    # otherwise do not crop.
                    if Config.data_collection['crop'] is not True:
                        image = image[Config.data_collection['image_crop_y1']:Config.data_collection['image_crop_y2'],
                                    Config.data_collection['image_crop_x1']:Config.data_collection['image_crop_x2']]

                    image = cv2.resize(image, 
                                    (config['input_image_width'],
                                    config['input_image_height']))
                    image = self.image_process.process(image)

                    images_timestep.append(image)
                    velocities_timestep.append(velocity)
                    #images_names_timestep.append(image_name)
                    
                    # if j is config['lstm_timestep']-1:
                    measurement = batch_samples[i][1][j]
                    steering_angle, throttle = measurement
                                                    
                    if abs(steering_angle) < config['steering_angle_jitter_tolerance']:
                        steering_angle = 0
                        
                    measurements_timestep.append(steering_angle*config['steering_angle_scale'])
                    throttles_timestep.append(throttle)
                    # data augmentation?
                    """
                    append, image, steering_angle = _data_augmentation(image, steering_angle)
                    if append is True:
                        images_timestep.append(image)
                        measurements_timestep.append(steering_angle*config['steering_angle_scale'])
                    """
                
                images.append(images_timestep)
                #images_names.append(images_names_timestep)
                measurements.append(measurements_timestep)
                velocities.append(velocities_timestep)
                throttles.append(throttles_timestep)
                # print(velocities)
            return images, throttles, measurements, velocities

        def _generator(samples, batch_size=config['batch_size']):
            num_samples = len(samples)
            while True: # Loop forever so the generator never terminates
               
                if config['lstm'] is True:
                    for offset in range(0, (num_samples//batch_size)*batch_size, batch_size):
                        batch_samples = samples[offset:offset+batch_size]

                        images, throttle, steering, velocities = _prepare_lstm_batch_samples(batch_samples)        
                        X_train_image = np.array(images)
                        X_train_velocity = np.array(velocities)
                        y_train = np.array(list(zip(steering,velocities)))
                        # print(measurements[0])
                        # print(y_train)
                        # reshape for lstm
                        X_train_image = X_train_image.reshape(-1, config['lstm_timestep'], 
                                        config['input_image_height'],
                                        config['input_image_width'],
                                        config['input_image_depth'])
                        X_train_velocity = X_train_velocity.reshape(-1, config['lstm_timestep'], 1)
                        
                        y_train = y_train.reshape(-1, config['lstm_timestep'], 2)
                        # print(X_train_image.shape)
                        # print(X_train_velocity.shape)
                        # print(y_train.shape)
                        # print(y_train[0])
                        yield [X_train_image, X_train_velocity], y_train

                else: 
                    samples = sklearn.utils.shuffle(samples)

                    for offset in range(0, num_samples, batch_size):
                        batch_samples = samples[offset:offset+batch_size]

                        images, measurements = _prepare_batch_samples(batch_samples)        

                        X_train = np.array(images)
                        y_train = np.array(measurements)

                        yield sklearn.utils.shuffle(X_train, y_train)
        
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
        from keras import losses, optimizers
        
        # checkpoint
        callbacks = []
        #weight_filename = self.data_path + '_' + Config.config_yaml_name \
        #    + '_N' + str(config['network_type']) + '_ckpt'
        checkpoint = ModelCheckpoint(self.model_ckpt_name +'.{epoch:02d}-{val_loss:.2f}.h5',
                                     monitor='val_loss', 
                                     verbose=1, save_best_only=True, mode='min')
        callbacks.append(checkpoint)
        
        # early stopping
        patience = config['early_stopping_patience']
        earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=patience, 
                                  verbose=1, mode='min')
        callbacks.append(earlystop)

        # tensor board
        logdir = config['tensorboard_log_dir'] + datetime.now().strftime("%Y%m%d-%H%M%S")
        tensorboard = TensorBoard(log_dir=logdir)
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
    def train(self, show_summary=True):
        
        self._prepare_data()
        self._build_model(show_summary)
        self._start_training()
        self.net_model.save(self.model_name)
        self._plot_training_history()
        Config.summary()
