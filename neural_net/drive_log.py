#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 22:07:31 2019
History:
11/28/2020: modified for OSCAR 

@author: jaerock
"""

import cv2
import numpy as np
#import keras
#import sklearn
#import resnet
from progressbar import ProgressBar
import matplotlib.pyplot as plt

import const
from net_model import NetModel
from drive_data import DriveData
from config import Config
from image_process import ImageProcess

config = Config.config

###############################################################################
#
class DriveLog:
    
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
        
        self.data_path = data_path
        self.drive = DriveData(csv_path)
        
        self.test_generator = None
        
        self.num_test_samples = 0        
        #self.config = Config()
        
        self.net_model = NetModel(model_path)
        self.net_model.load()
        self.model_path = model_path
        
        self.image_process = ImageProcess()


    ###########################################################################
    #
    def _prepare_data(self):
        
        self.drive.read()
    
        self.test_data = list(zip(self.drive.image_names, self.drive.measurements))
        self.num_test_samples = len(self.test_data)
        
        print('Test samples: {0}'.format(self.num_test_samples))
    
   ###########################################################################
    #
    def _plot_result(self):
        fig, (ax1, ax2) = plt.subplots(1, 2)

        # Plot a histogram of the prediction errors
        num_bins = 25
        hist, bins = np.histogram(diffs, num_bins)
        center = (bins[:-1]+ bins[1:]) * 0.5
        ax1.bar(center, hist, width=0.05)
        ax1.title('Historgram of Predicted Error')
        ax1.xlabel('Steering Angle')
        ax1.ylabel('Number of predictions')
        ax1.xlim(-1.0, 1.0)
        ax1.plot(np.min(diffs), np.max(diffs))
        ax1.savefig(self.model_path + '_err_hist.png')

        # Plot a Scatter Plot of the Error
        ax2.scatter(mesus, preds)
        ax2.xlabel('True Values ')
        ax2.ylabel('Predictions ')
        ax2.axis('equal')
        ax2.axis('square')
        ax2.xlim([-1.0, 1.0])
        ax2.ylim([-1.0, 1.0])
        ax2.plot([-1.0, 1.0], [-1.0, 1.0], color='k', linestyle='-', linewidth=.1)
        ax2.savefig(self.model_path + '_scatter.png')
        
        plt.show()


   ###########################################################################
    #
    def run(self):
        
        self._prepare_data()
        #fname = self.data_path + const.LOG_EXT
        fname = self.model_path + const.LOG_EXT # use model name to save log
        
        file = open(fname, 'w')

        #print('image_name', 'label', 'predict', 'abs_error')
        bar = ProgressBar()

        mesus = []
        preds = []
        diffs = []
        
        file.write('image_name, label, predict, abs_error\n')
        for image_name, measurement in bar(self.test_data):   
            image_fname = self.data_path + '/' + image_name
            image = cv2.imread(image_fname)
            image = cv2.resize(image, (Config.config['input_image_width'],
                                       Config.config['input_image_height']))
            image = self.image_process.process(image)
            
            npimg = np.expand_dims(image, axis=0)
            predict = self.net_model.model.predict(npimg)
            predict = predict / Config.config['steering_angle_scale']
            
            mesus.append(measurement[0])
            preds.append(predict[0][0])
            diffs.append(measurement[0]-predict[0][0])
            #print(image_name, measurement[0], predict[0][0],\ 
            #                  abs(measurement[0]-predict[0][0]))
            if Config.config['lstm'] is True:
                log = image_name+','+str(measurement[0])+','+str(predict[0][0][0])\
                                +','+str(abs(measurement[0]-predict[0][0][0]))
            else:
                log = image_name+','+str(measurement[0])+','+str(predict[0][0])\
                                +','+str(abs(measurement[0]-predict[0][0]))

            file.write(log+'\n')
        
        file.close()
        print(fname + ' created.')

        _plot_result(mesus, preds, diffs)
