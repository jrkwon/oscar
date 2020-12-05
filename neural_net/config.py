#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 13:23:14 2017
History:
11/28/2020: modified for OSCAR 

@author: jaerock
"""

###############################################################################
#

import os
import yaml

class Config:
    config_name = os.environ['OSCAR_PATH'] + '/config/neural_net/' + 'config.yaml'
    with open(config_name) as file:
        config_yaml = yaml.load(file, Loader=yaml.FullLoader)
        config_yaml_name = config_yaml['config_yaml']

    yaml_name = os.environ['OSCAR_PATH'] + '/config/neural_net/' + config_yaml_name + '.yaml'
    with open(yaml_name) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    def __init__(self): # model_name):
        pass
    
    @staticmethod
    def summary():
        print('============ neural_net configuration ============')
        print('+ config name: ' + Config.config_yaml_name + '.yaml')
        print('+ training ---------------------------------------')
        print('-- network_type: ' + str(Config.config['network_type']))
        print('-- lstm: ' + str(Config.config['lstm']))
        print('-- validation_rate: ' + str(Config.config['validation_rate']))
        print('-- num_epochs: ' + str(Config.config['num_epochs']))
        print('-- batch_size: ' + str(Config.config['batch_size']))
        print('-- num_outputs: ' + str(Config.config['num_outputs']))
        print('+ steering angle preprocessing -------------------')
        print('-- steering_angle_scale: ' + str(Config.config['steering_angle_scale']))
        print('-- steering_angle_jitter_tolerance: '
              + str(Config.config['steering_angle_jitter_tolerance']))
        print('+ data augmentation ------------------------------')
        print('-- data_aug_flip: ' + str(Config.config['data_aug_flip']))
        print('-- data_aug_bright: ' + str(Config.config['data_aug_bright']))
        print('-- data_aug_shift: ' + str(Config.config['data_aug_shift']))
        print('-- image_size: ' + str(Config.config['input_image_width']) + 'x' \
              + str(Config.config['input_image_height']))
        print('+ data collection --------------------------------')
        print('-- camera image topic: ' + Config.config['camera_image_topic'])
        print('-- capture_area : ({0}, {1}) - ({2}, {3})'.format(
                                  Config.config['image_crop_x1'],
                                  Config.config['image_crop_y1'],
                                  Config.config['image_crop_x2'],
                                  Config.config['image_crop_y2']))
        
if __name__ == '__main__':
    Config.summary()