#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 13:49:23 2017
History:
11/28/2020: modified for OSCAR 

@author: jaerock
"""


import sys
from drive_train import DriveTrain
from config import Config

###############################################################################
#
def train(data_folder_name, load_model_name=None):
    drive_train = DriveTrain(data_folder_name)
    drive_train.train(show_summary=False, load_model_name=load_model_name)


###############################################################################
#
if __name__ == '__main__':
    try:
        if Config.neural_net['load_weight'] is True:
            if (len(sys.argv) != 3):
                exit('Usage:\n$ python {} data_path load_model_name'.format(sys.argv[0]))
            train(sys.argv[1], sys.argv[2])
        
        else:
            if (len(sys.argv) != 2):
                exit('Usage:\n$ python {} data_path'.format(sys.argv[0]))
            train(sys.argv[1])

    except KeyboardInterrupt:
        print ('\nShutdown requested. Exiting...')
