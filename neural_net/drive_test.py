#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 13:23:14 2017

@author: jaerock
"""

import cv2
import numpy as np
#import keras
import sklearn
from progressbar import ProgressBar

#import resnet
import const
from net_model import NetModel
from drive_data import DriveData
from config import Config
from image_process import ImageProcess

config = Config.neural_net

###############################################################################
#
class DriveTest:
    
    ###########################################################################
    # model_path = 'path_to_pretrained_model_name' excluding '.h5' or 'json'
    # data_path = 'path_to_drive_data'  e.g. ../data/2017-09-22-10-12-34-56'
    def __init__(self, model_path, data_path):
        if data_path[-1] == '/':
            data_path = data_path[:-1]

        loc_slash = data_path.rfind('/')
        if loc_slash != -1: # there is '/' in the data path
            model_name = data_path[loc_slash+1:] # get folder name
            #model_name = model_name.strip('/')
        else:
            model_name = data_path
        csv_path = data_path + '/' + model_name + const.DATA_EXT   
        
        self.drive = DriveData(csv_path)
        
        self.test_generator = None
        
        self.num_test_samples = 0        
        #self.config = Config()
        
        self.net_model = NetModel(model_path)
        self.net_model.load()
        
        self.image_process = ImageProcess()
        self.data_path = data_path

      
    ###########################################################################
    #
    def _prepare_data(self):
    
        self.drive.read()
        
        samples = list(zip(self.drive.image_names, self.drive.measurements))

        if config['lstm'] is True:
            self.test_data = self._prepare_lstm_data(samples)
        else:    
            self.test_data = samples
        
        self.num_test_samples = len(self.test_data)
        
        print('Test samples: ', self.num_test_samples)
    
                                          
    ###########################################################################
    # group the samples by the number of timesteps
    def _prepare_lstm_data(self, samples):
        num_samples = len(samples)

        # get the last index number      
        steps = 1
        last_index = (num_samples - config['lstm_timestep'])//steps
        
        image_names = []
        measurements = []
        for i in range(0, last_index, steps):
            sub_samples = samples[ i : i+config['lstm_timestep'] ]
            
            # print('num_batch_sample : ',len(batch_samples))
            sub_image_names = []
            sub_measurements = []
            for image_name, measurment in sub_samples:
                sub_image_names.append(image_name)
                sub_measurements.append(measurment)

            image_names.append(sub_image_names)
            measurements.append(sub_measurements)
        
        return list(zip(image_names, measurements))


    ###########################################################################
    #
    def _prep_generator(self):
        
        if self.data_path == None:
            raise NameError('data_path must be set.')
            
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
                
                ## data augmentation <-- doesn't need since this is not training
                #append, image, steering_angle = _data_augmentation(image, steering_angle)
                #if append is True:
                #    images.append(image)
                #    measurements.append(steering_angle*config['steering_angle_scale'])

            return images, measurements
            
        def _prepare_lstm_batch_samples(batch_samples):
            images = []
            measurements = []

            for i in range(0, config['batch_size']):

                images_timestep = []
                #images_names_timestep = []
                measurements_timestep = []

                for j in range(0, config['lstm_timestep']):

                    image_name = batch_samples[i][0][j]
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

                    images_timestep.append(image)
                    #images_names_timestep.append(image_name)
                    
                    if j is config['lstm_timestep']-1:
                        measurement = batch_samples[i][1][j]
                        steering_angle, throttle = measurement
                                                    
                        if abs(steering_angle) < config['steering_angle_jitter_tolerance']:
                            steering_angle = 0
                            
                        measurements_timestep.append(steering_angle*config['steering_angle_scale'])

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

            return images, measurements

        def _generator(samples, batch_size=config['batch_size']):
            num_samples = len(samples)
            while True: # Loop forever so the generator never terminates
                
                bar = ProgressBar()
                
                if config['lstm'] is True:
                    for offset in bar(range(0, (num_samples//batch_size)*batch_size, batch_size)):
                        batch_samples = samples[offset:offset+batch_size]

                        images, measurements = _prepare_lstm_batch_samples(batch_samples)        

                        X_train = np.array(images)
                        y_train = np.array(measurements)

                        # reshape for lstm
                        X_train = X_train.reshape(-1, config['lstm_timestep'], 
                                        config['input_image_height'],
                                        config['input_image_width'],
                                        config['input_image_depth'])
                        y_train = y_train.reshape(-1, 1)

                        yield X_train, y_train

                else: 
                    samples = sklearn.utils.shuffle(samples)

                    for offset in bar(range(0, num_samples, batch_size)):
                        batch_samples = samples[offset:offset+batch_size]

                        images, measurements = _prepare_batch_samples(batch_samples)        

                        X_train = np.array(images)
                        y_train = np.array(measurements)

                        yield sklearn.utils.shuffle(X_train, y_train)     

        self.test_generator = _generator(self.test_data)
        
    
    ###########################################################################
    #
    def _start_test(self):

        if (self.test_generator == None):
            raise NameError('Generators are not ready.')
        
        print('Evaluating the model with test data sets ...')
        ## Note: Do not use multiprocessing or more than 1 worker.
        ##       This will genereate threading error!!!
        score = self.net_model.model.evaluate_generator(self.test_generator, 
                                self.num_test_samples//config['batch_size']) 
                                #workers=1)
        print('Loss: {0}'.format(score)) #[0], "Accuracy: ", score[1])
        #print("\nLoss: ", score[0], "rmse: ", score[1])
        
    

   ###########################################################################
    #
    def test(self):
        self._prepare_data()
        self._prep_generator()
        self._start_test()
        Config.summary()

