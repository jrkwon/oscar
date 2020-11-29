#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 13:23:14 2017
History:
11/28/2020: modified for OSCAR 

@author: jaerock
"""

import cv2
import numpy as np

#from config import Config

class DataAugmentation():
    def __init__(self):
#        self.config = Config()
        self.bright_limit = (-0.5, 0.15)
        self.shift_range = (40,5)
        self.brightness_multiplier = None
        self.image_hsv = None
        self.rows = None
        self.cols = None
        self.ch = None
        self.shift_x = None
        self.shift_y = None
        self.shift_matrix = None

    def flipping(self, img, steering):
        flip_image = cv2.flip(img,1)
        flip_steering = steering*-1.0
        return flip_image, flip_steering

    def brightness(self, img):
        self.brightness_multiplier = 1.0 + np.random.uniform(low=self.bright_limit[0], high=self.bright_limit[1])
        self.image_hsv = cv2.cvtColor(img, cv2.COLOR_RGB2HSV)
        self.image_hsv[:,:,2] = self.image_hsv[:,:,2] * self.brightness_multiplier
        bright_image = cv2.cvtColor(self.image_hsv, cv2.COLOR_HSV2RGB)
        return bright_image

    def shift(self, img, steering):
        self.rows, self.cols, self.ch = img.shape
        self.shift_x = self.shift_range[0]*np.random.uniform()-self.shift_range[0]/2
        #print (self.shift_x)
        shift_steering = steering + (self.shift_x/self.shift_range[0]*2*0.2) * -1
        #print (shift_steering)
        self.shift_y = self.shift_range[1]*np.random.uniform()-self.shift_range[1]/2
        #print (self.shift_y)
        self.shift_matrix = np.float32([[1,0,self.shift_x],[0,1,self.shift_y]])
        shift_image = cv2.warpAffine(img, self.shift_matrix, (self.cols, self.rows))
        return shift_image, shift_steering
