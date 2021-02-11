#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 15:07:31 2021
History:
2/10/2021: modified for OSCAR 

@author: jaerock
"""


import sys

from drive_view import DriveView


###############################################################################
#       
def main(weight_name, data_folder_name, target_folder_name):
    drive_view = DriveView(weight_name, data_folder_name, target_folder_name) 
    drive_view.run() # data folder path to test
       

###############################################################################
#       
if __name__ == '__main__':
    try:
        if (len(sys.argv) == 3):
            main(None, sys.argv[1], sys.argv[2])
        elif (len(sys.argv) == 4):
            main(sys.argv[1], sys.argv[2], sys.argv[3])
        else:
            msg1 = 'Usage:\n$ python {}.py weight_name data_folder_name target_folder_name'.format(sys.argv[0]) 
            msg2 = '\n$ python {}.py data_folder_name target_folder_name'.format(sys.argv[0]) 
            msg = 'Use either of followings\n' + msg1 + msg2
            exit(msg)
        
    except KeyboardInterrupt:
        print ('\nShutdown requested. Exiting...')
