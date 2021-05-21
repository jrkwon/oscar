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
import cv2
import numpy as np
from vis.utils import utils
from vis.visualization import visualize_cam

from drive_run import DriveRun
from drive_data import DriveData
from config import Config
from image_process import ImageProcess


###############################################################################
#       
def images_saliency(model_path, image_folder_path):
    image_process = ImageProcess()
    image_folder_name = None
    if image_folder_path[-1] == '/':
        image_folder_path = image_folder_path[:-1]
    loc_slash = image_folder_path.rfind('/')
    if loc_slash != -1: # there is '/' in the data path
        image_folder_name = image_folder_path[loc_slash+1:] 
    csv_path = image_folder_path+'/'+image_folder_name+'.csv'
    
    data = DriveData(csv_path)
    data.read(normalize=False)
    
    images_name = data.image_names
    steering_angles = data.measurements
    
    drive_run = DriveRun(model_path)
    if os.path.isdir(image_folder_path + '/saliency'+ '_' + model_path[-2:]) is not True:
        os.mkdir(image_folder_path + '/saliency'+ '_' + model_path[-2:])
    
    # image_process = ImageProcess()
    for i in range(len(images_name)):
        image_file_path = image_folder_path+'/'+images_name[i]
        image = cv2.imread(image_file_path)
        if Config.data_collection['crop'] is not True:
            image = image[Config.data_collection['image_crop_y1']:Config.data_collection['image_crop_y2'],
                        Config.data_collection['image_crop_x1']:Config.data_collection['image_crop_x2']]
        image = cv2.resize(image, 
                            (Config.neural_net['input_image_width'],
                            Config.neural_net['input_image_height']))
        image = image_process.process(image, bgr=True)
        measurement = drive_run.run((image, ))
        
        fig = plt.figure()
        # plt.title('Prediction:\t' + str(measurement[0][0]) + '\nGroundTruth:\t' + str(steering_angles[i][0])
        #           + '\nError:\t' + str(abs(steering_angles[i][0] - measurement[0][0])))
        layer_idx = utils.find_layer_idx(drive_run.net_model.model, 'maxpool_3')
        # penultimate_layer_idx = utils.find_layer_idx(drive_run.net_model.model, 'conv2d_4')
        fc_last = utils.find_layer_idx(drive_run.net_model.model, 'fc_str')
        # fc_last = utils.find_layer_idx(drive_run.net_model.model, 'dense_5')
        heatmap = visualize_cam(drive_run.net_model.model, layer_idx, 
                    filter_indices=None, seed_input=image, backprop_modifier='guided', penultimate_layer_idx=None)

        ax1 = fig.add_subplot(2,1,1)
        ax1.set_title('Prediction :' + str(format(measurement[0][0], ".9f")) 
                  + '\nGroundTruth:' + str(format(steering_angles[i][0], ".9f"))
                  + '\nError      :' + str(format(abs(steering_angles[i][0]-measurement[0][0]), ".9f"))
                  )
        plt.imshow(image)
        plt.imshow(heatmap, cmap='jet', alpha=0.5)   
        ax2 = fig.add_subplot(2,1,2) 
        plt.subplot(2,1,2)
        plt.imshow(image)
                
        saliency_file_path = image_folder_path + '/saliency' + '_' + model_path[-2:] + '/' + images_name[i][:-4] + '_saliency.png'
        plt.tight_layout()
        # save fig    
        plt.savefig(saliency_file_path, dpi=150)
        
        cur_output = '{0}/{1}\r'.format(i, len(images_name))

        sys.stdout.write(cur_output)
        sys.stdout.flush()
        # print('Saved ' + saliency_file_path)
        # print(image_path)
    # measurement = drive_run.run((image, ))
    
    # print("csv",csv_path)

def show_layer_saliency(model_path, image_folder_path):
    image_process = ImageProcess()
    image_folder_name = None
    if image_folder_path[-1] == '/':
        image_folder_path = image_folder_path[:-1]
    loc_slash = image_folder_path.rfind('/')
    if loc_slash != -1: # there is '/' in the data path
        image_folder_name = image_folder_path[loc_slash+1:] 
    csv_path = image_folder_path+'/'+image_folder_name+'.csv'
    
    dir_file_list = os.listdir(image_folder_path)
    dir_img_list = [img for img in dir_file_list if img.endswith(".png") or img.endswith(".jpg")]
    
    print(dir_img_list)
    
    drive_run = DriveRun(model_path)
    if os.path.isdir(image_folder_path + '/saliency'+ '_' + model_path[-2:]) is not True:
        os.mkdir(image_folder_path + '/saliency'+ '_' + model_path[-2:])
    
    num_conv_layer = 0
    for i in range(len(drive_run.net_model.model.layers)):
            if "Conv2D" in str(drive_run.net_model.model.layers[i]):
                num_conv_layer += 1
    
    
    for j in range(len(dir_img_list)):
        first_layer = True
        fig = plt.figure(figsize=(3,10))
        # fig, ax = plt.subplots(1, num_conv_layer-1)
        
        num_layer = 1
        for i in range(len(drive_run.net_model.model.layers)):
            if "Conv2D" in str(drive_run.net_model.model.layers[i]):
                if first_layer is True:
                    first_layer = False
                else:
                    # print(drive_run.net_model.model.get_layer(index = i).name)
                    
                    image_file_path = image_folder_path+'/'+dir_img_list[j]
                    image = cv2.imread(image_file_path)
                    
                    if Config.data_collection['crop'] is not True:
                        image = image[Config.data_collection['image_crop_y1']:Config.data_collection['image_crop_y2'],
                                    Config.data_collection['image_crop_x1']:Config.data_collection['image_crop_x2']]
                    image = cv2.resize(image, 
                                        (Config.neural_net['input_image_width'],
                                        Config.neural_net['input_image_height']))
                    image = image_process.process(image, bgr=False)
                    
                    layer_idx = utils.find_layer_idx(drive_run.net_model.model, drive_run.net_model.model.get_layer(index = i).name)
                    penultimate_layer_idx = utils.find_layer_idx(drive_run.net_model.model, drive_run.net_model.model.get_layer(index = i-1).name)
                    fc_last = utils.find_layer_idx(drive_run.net_model.model, 'dense_5')
                    # fc_last = utils.find_layer_idx(drive_run.net_model.model, 'fc_str')
                    heatmap = visualize_cam(drive_run.net_model.model, layer_idx, 
                                filter_indices=fc_last, seed_input=image, backprop_modifier='guided', penultimate_layer_idx=penultimate_layer_idx)

                    # ax[num_layer-1].imshow(image)
                    # ax[num_layer-1].imshow(heatmap, cmap='jet', alpha=0.5)
                    ax = fig.add_subplot(num_conv_layer-1,1, num_layer)
                    # ax[num_layer-1].set_axis_off()
                    plt.imshow(image)
                    plt.imshow(heatmap, cmap='jet', alpha=0.5)
                    num_layer += 1
        plt.subplots_adjust(wspace=0, hspace=1)
        plt.tight_layout()
        saliency_file_path = image_folder_path + '/saliency' + '_' + model_path[-2:] + '/' + dir_img_list[j][:-4] + '_saliency_layers.png'
        # save fig    
        plt.savefig(saliency_file_path, dpi=150)
        
        cur_output = '{0}/{1}\r'.format(j+1, len(dir_img_list))

        sys.stdout.write(cur_output)
        sys.stdout.flush()
        del fig

def main(model_path, image_file_path):
    image_process = ImageProcess()

    image = cv2.imread(image_file_path)

    # if collected data is not cropped then crop here
    # otherwise do not crop.
    if Config.data_collection['crop'] is not True:
        image = image[Config.data_collection['image_crop_y1']:Config.data_collection['image_crop_y2'],
                      Config.data_collection['image_crop_x1']:Config.data_collection['image_crop_x2']]

    image = cv2.resize(image, 
                        (Config.neural_net['input_image_width'],
                         Config.neural_net['input_image_height']))
    image = image_process.process(image)

    drive_run = DriveRun(model_path)
    measurement = drive_run.run((image, ))

    """ grad modifier doesn't work somehow
    fig, axs = plt.subplots(1, 3)
    fig.suptitle('Saliency Visualization' + str(measurement))
    titles = ['left steering', 'right steering', 'maintain steering']
    modifiers = [None, 'negate', 'small_values']

    for i, modifier in enumerate(modifiers):
        layer_idx = utils.find_layer_idx(drive_run.net_model.model, 'conv2d_last')
        heatmap = visualize_cam(drive_run.net_model.model, layer_idx, 
                    filter_indices=None, seed_input=image, backprop_modifier='guided', 
                    grad_modifier=modifier)

        axs[i].set(title = titles[i])
        axs[i].imshow(image)
        axs[i].imshow(heatmap, cmap='jet', alpha=0.3)
    """
    plt.figure()
    #plt.title('Saliency Visualization' + str(measurement))
    plt.title('Steering Angle Prediction: ' + str(measurement[0][0]))
    layer_idx = utils.find_layer_idx(drive_run.net_model.model, 'conv2d_last')
    heatmap = visualize_cam(drive_run.net_model.model, layer_idx, 
                filter_indices=None, seed_input=image, backprop_modifier='guided')

    # plt.imshow(image)
    # plt.imshow(heatmap, cmap='jet', alpha=0.5)

    # file name
    loc_slash = image_file_path.rfind('/')
    if loc_slash != -1: # there is '/' in the data path
        image_file_name = image_file_path[loc_slash+1:] 

    saliency_file_path = model_path + '_' + image_file_name + '_saliency.png'
    saliency_file_path_pdf = model_path + '_' + image_file_name + '_saliency.pdf'

    plt.tight_layout()
    # save fig    
    plt.savefig(saliency_file_path, dpi=150)
    plt.savefig(saliency_file_path_pdf, dpi=150)

    print('Saved ' + saliency_file_path +' & .pdf')

    # show the plot 
    #plt.show()

###############################################################################
#       
if __name__ == '__main__':
    try:
        if (len(sys.argv) != 3):
            exit('Usage:\n$ python {} model_path, image_file_name'.format(sys.argv[0]))

        # main(sys.argv[1], sys.argv[2])
        
        images_saliency(sys.argv[1], sys.argv[2])
        # show_layer_saliency(sys.argv[1], sys.argv[2])

    except KeyboardInterrupt:
        print ('\nShutdown requested. Exiting...')
