#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 13:49:23 2017
History:
11/28/2020: modified for OSCAR 

@author: jaerock
"""


import sys
from drive_train_anec import DriveTrain
import gpu_options

###############################################################################
#
def train(model_type, data_folder_name):
    gpu_options.set()
    
    drive_train = DriveTrain(model_type, data_folder_name)
    drive_train.train(show_summary = False)


###############################################################################
#
if __name__ == '__main__':
    try:
        if (len(sys.argv) != 3):
            exit('Usage:\n$ python {} data_path'.format(sys.argv[0]))

        # sys.argv[1] --> model type --> BM or DCM 
        # sys.argv[2] --> data folder
        if sys.argv[1] == 'BM' or sys.argv[1] == 'DCM':
            train(sys.argv[1], sys.argv[2])
        else:
            exit('Please enter valid model type; BM or DCM')
        

    except KeyboardInterrupt:
        print ('\nShutdown requested. Exiting...')
