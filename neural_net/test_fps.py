#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 13:49:23 2017
History:
11/28/2020: modified for OSCAR 

@author: jaerock, donghyun
"""

import sys, os
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2, time
import numpy as np
from vis.utils import utils
from vis.visualization import visualize_cam

from drive_run import DriveRun
from drive_data import DriveData
from config import Config
from image_process import ImageProcess


###############################################################################
#       
def images_fps(model_path, image_folder_path):
    image_process = ImageProcess()
    if image_folder_path[-1] != '/':
        image_folder_path = image_folder_path + '/'
    
    drive_run = DriveRun(model_path)
    # if os.path.isdir(image_folder_path + '/fps'+ '_' + model_path[-2:]) is not True:
    #     os.mkdir(image_folder_path + '/fps'+ '_' + model_path[-2:])
    
    images_name = os.listdir(image_folder_path)
    print(len(images_name))
    # image_process = ImageProcess()
    total_time = 0
    for i in range(len(images_name)):
        image_file_path = image_folder_path+images_name[i]
        image = cv2.imread(image_file_path)
        if Config.data_collection['crop'] is not True:
            image = image[Config.data_collection['image_crop_y1']:Config.data_collection['image_crop_y2'],
                        Config.data_collection['image_crop_x1']:Config.data_collection['image_crop_x2']]
        image = cv2.resize(image, 
                            (Config.neural_net['input_image_width'],
                            Config.neural_net['input_image_height']))
        image = image_process.process(image, bgr=True)
        start = time.time()
        measurement = drive_run.run((image, ))
        # print(i, measurement[0][0])
        end = time.time() - start
        total_time += end
        # print(i, end)
    print('total_time : ', total_time)
    
def images_lstm_fps(model_path, image_folder_path):
    image_process = ImageProcess()
    if image_folder_path[-1] != '/':
        image_folder_path = image_folder_path + '/'
    
    drive_run = DriveRun(model_path)
    # if os.path.isdir(image_folder_path + '/fps'+ '_' + model_path[-2:]) is not True:
    #     os.mkdir(image_folder_path + '/fps'+ '_' + model_path[-2:])
    
    images_name = os.listdir(image_folder_path)
    print(len(images_name))
    # image_process = ImageProcess()
    total_time = 0
    
    image_file_path = image_folder_path+images_name[0]
    image = cv2.imread(image_file_path)
    if Config.data_collection['crop'] is not True:
        image = image[Config.data_collection['image_crop_y1']:Config.data_collection['image_crop_y2'],
                    Config.data_collection['image_crop_x1']:Config.data_collection['image_crop_x2']]
    image = cv2.resize(image, 
                        (Config.neural_net['input_image_width'],
                        Config.neural_net['input_image_height']))
    image = image_process.process(image, bgr=True)
    
    images = []
    for i in range(Config.neural_net['lstm_timestep']):
        images.append(image)
    
    for _ in range(100):
        start = time.time()
        measurement = drive_run.run((images, ))
        # print(i, measurement[0][0])
        end = time.time() - start
        total_time += end
    # print(i, end)
    print('total_time : ', total_time)
        
        
if __name__ == '__main__':
    try:
        if (len(sys.argv) != 3):
            exit('Usage:\n$ python {} model_path, image_file_name'.format(sys.argv[0]))

        # main(sys.argv[1], sys.argv[2])
        if Config.neural_net['lstm'] is False:
            images_fps(sys.argv[1], sys.argv[2])
        else:
            images_lstm_fps(sys.argv[1], sys.argv[2])
            

    except KeyboardInterrupt:
        print ('\nShutdown requested. Exiting...')
