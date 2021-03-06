#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 13:49:23 2017
History:
11/28/2020: modified for OSCAR 

@author: jaerock
"""


import sys

from drive_test import DriveTest
    

###############################################################################
#       
def test(trained_model_name, data_path):
    drive_test = DriveTest(trained_model_name, data_path)
    drive_test.test()    
       

###############################################################################
#       
if __name__ == '__main__':
    try:
        if (len(sys.argv) != 3):
            exit('Usage:\n$ python {} model_name data_path'.format(sys.argv[0]))

        test(sys.argv[1], sys.argv[2])

    except KeyboardInterrupt:
        print ('\nShutdown requested. Exiting...')
