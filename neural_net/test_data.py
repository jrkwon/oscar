#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 13:49:23 2017
History:
11/28/2020: modified for OSCAR 

@author: jaerock
"""


import sys
import numpy as np
import const
from drive_data import DriveData
from config import Config

config = Config.config


###############################################################################
#       
def main(data_path):
    if data_path[-1] == '/':
        data_path = data_path[:-1]

    loc_slash = data_path.rfind('/')
    if loc_slash != -1: # there is '/' in the data path
        model_name = data_path[loc_slash + 1:] # get folder name
        #model_name = model_name.strip('/')
    else:
        model_name = data_path
    csv_path = data_path + '/' + model_name + const.DATA_EXT   
    
    data = DriveData(csv_path)
    data.read(normalize_data = config['normalize_data'], read = False)


###############################################################################
#       
if __name__ == '__main__':
    try:
        if (len(sys.argv) != 2):
            exit('Usage:\n$ python test_data.py data_path')

        main(sys.argv[1])

    except KeyboardInterrupt:
        print ('\nShutdown requested. Exiting...')
