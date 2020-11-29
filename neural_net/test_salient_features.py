#!/usr/bin/env python3
# -*- coding: utf-8 -*-
'''
Created on Fri May 17 3:13:00 2019

@author: ninad
@author: jaerock
'''

import numpy as np
import sys
import os
import pandas as pd
from progressbar import ProgressBar
from scipy import misc
from drive_run import DriveRun
from config import Config
from image_process import ImageProcess

#sys.path.append('/home/ghor9797/NCD_Guthub/python/keras-vis')
#sys.path.append(str(os.environ['HOME']) + ('/keras-vis'))
from vis.utils import utils
from vis.visualization import visualize_saliency, overlay

###############################################################################
#
def test_salient_features(data_path, output_path):

    drive = DriveRun(data_path)
    config = Config.config

    # add '/' at the end of data_path if user doesn't specify
    if data_path[-1] != '/':
        data_path = data_path + '/'

    # find the second '/' from the end to get the folder name
    loc_dir_delim = data_path[:-1].rfind('/')
    if (loc_dir_delim != -1):
        folder_name = data_path[loc_dir_delim+1:-1]
        csv_file = folder_name + const.DATA_EXT
    else:
        folder_name = data_path[:-1]
        csv_file = folder_name + const.DATA_EXT


    csv_header = ['image_fname', 'steering_angle', 'throttle']
    df = pd.read_csv(csv_file, names=csv_header, index_col=False)
    num_data = len(df)
    text = open(output_path + '/salient.txt', 'w+')
    bar = ProgressBar()
    image_process = ImageProcess()

    for i in bar(range(num_data)):
        image_name = df.loc[i]['image_fname']
        steering = df.loc[i]['steering_angle']
        image_path = data_path + image_name + '.jpg'
        image = utils.load_img(image_path, target_size=(config.image_size[1],
                                                        config.image_size[0]))
        image = image_process.process(image)
        prediction = drive.run(image)
        text.write(str(image_name) + '\t' + str(steering) + '\t' + str(prediction))
        
        modifiers = [None, 'negate', 'small_values']
        for i, modifier in enumerate(modifiers):
            heatmap = visualize_saliency(drive.net_model.model, layer_idx=-1,
                                         filter_indices=0, seed_input=image,
                                         grad_modifier=modifier, keepdims=True)
            final = overlay(image, heatmap, alpha=0.5)
            cv2.imwrite(output_path + '/' + image_name + '_' + str(i) + '.jpg', final)


###############################################################################
#
def main():
    if (len(sys.argv) != 3):
        print('Usage: \n$ test_salient_features data_folder_name output_folder_name')
        return

    test_salient_features(sys.argv[1])


###############################################################################
#
if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nShutdown requested. Exiting...')
