#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 13:23:14 2017
History:
11/28/2020: modified for OSCAR 

@author: jaerock
"""

#####
# We don't need this cropping process anymore
#####

from PIL import Image
import os, sys
import const
from config import Config

###############################################################################
#       
def list_files(directory, extension):
    return (f for f in os.listdir(directory) if f.endswith(extension))

###############################################################################
#       
def main(data_path):
    #config = Config()
    dirs = list_files(data_path, const.IMAGE_EXT)

    for item in dirs:
        fullpath = os.path.join(data_path,item)         #corrected
        if os.path.isfile(fullpath):
            im = Image.open(fullpath)
            fname, ext = os.path.splitext(fullpath)
            cropped = im.crop((Config.data_collection['image_crop_x1'], Config.data_collection['image_crop_y1'],
                              Config.data_collection['image_crop_x2'], Config.data_collection['image_crop_y2'])) 
            cropped.save(fname + '_crop' + ext)
            print('Cropped - ' + fname + ext)



###############################################################################
#       
if __name__ == '__main__':
    try:
        if (len(sys.argv) != 2):
            exit('Usage:\n$ image_crop data_path')
        
        main(sys.argv[1])

    except KeyboardInterrupt:
        print ('\nShutdown requested. Exiting...')
       
