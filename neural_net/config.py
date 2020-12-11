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
import sys

class Config:
    try:
        config_name = os.environ['OSCAR_PATH'] + '/config/neural_net/' + 'config.yaml'
    except:
        exit('ERROR: OSCAR_PATH not defined. Please source setup.bash.') 

    with open(config_name) as file:
        config_yaml = yaml.load(file, Loader=yaml.FullLoader)
        config_yaml_name = config_yaml['config_yaml']
        print('=======================================================')
        print('\tConfiguration Settings \n\t' + yaml.dump(config_yaml))

    yaml_name = os.environ['OSCAR_PATH'] + '/config/neural_net/' + config_yaml_name + '.yaml'
    with open(yaml_name) as file:
        config = yaml.load(file, Loader=yaml.FullLoader)

    def __init__(self): # model_name):
        pass
    
    @staticmethod
    def summary():
        print('=======================================================')
        print('                 System Configuration ')
        print('-------------------------------------------------------')
        print(yaml.dump(Config.config))


if __name__ == '__main__':
    Config.summary()