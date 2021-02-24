#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 13:23:14 2017
History:
11/28/2020: modified for OSCAR 

@author: jaerock
"""

import numpy as np
from net_model import NetModel
from config import Config

###############################################################################
#
class DriveRun:
    
    ###########################################################################
    # model_path = 'path_to_pretrained_model_name' excluding '.h5' or 'json'
    # data_path = 'path_to_drive_data'  e.g. ../data/2017-09-22-10-12-34-56'
    def __init__(self, model_path):
        
        #self.config = Config()
        self.net_model = NetModel(model_path)   
        self.net_model.load()

   ###########################################################################
    #
    def run(self, image, vel):
        npimg = np.expand_dims(image, axis=0)
        npimg = np.array(npimg).reshape(-1, 
                                          160,
                                          160,
                                          3)
        vel = np.array(vel).reshape(-1, 1)
        # X_train = np.stack([X_train_img, X_train_vel], axis=1)
        X_train = [npimg, vel]
        measurements, throttle = self.net_model.model.predict(X_train)
        measurements = measurements / Config.neural_net['steering_angle_scale']
        return measurements, throttle
