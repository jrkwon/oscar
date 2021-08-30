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
from PIL import Image
from vis.utils import utils
from vis.visualization import visualize_cam, visualize_saliency
from keras.models import Sequential, Model

from drive_run import DriveRun
from drive_data import DriveData
from config import Config
from image_process import ImageProcess


###############################################################################
#
def images_cam(model_path, image_folder_path):
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
        image = image_process.process(image, bgr=False)
        measurement = drive_run.run((image, ))
        
        fig = plt.figure()
        # plt.title('Prediction:\t' + str(measurement[0][0]) + '\nGroundTruth:\t' + str(steering_angles[i][0])
        #           + '\nError:\t' + str(abs(steering_angles[i][0] - measurement[0][0])))
        layer_idx = utils.find_layer_idx(drive_run.net_model.model, 'conv_4')
        # penultimate_layer_idx = utils.find_layer_idx(drive_run.net_model.model, 'conv2d_4')
        fc_last = utils.find_layer_idx(drive_run.net_model.model, 'fc_str')
        # fc_last = utils.find_layer_idx(drive_run.net_model.model, 'fc_1')
        heatmap = visualize_cam(drive_run.net_model.model, layer_idx, filter_indices=fc_last, seed_input=image, backprop_modifier='guided')

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
def images_saliency(model_path, image_folder_path, islstm):
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
    
    if islstm is True:
        images = []
        for i in range(len(images_name)):
            image_file_path = image_folder_path+'/'+images_name[i]
            # print(image_file_path)
            image = cv2.imread(image_file_path)
            if Config.data_collection['crop'] is not True:
                image = image[Config.data_collection['image_crop_y1']:Config.data_collection['image_crop_y2'],
                            Config.data_collection['image_crop_x1']:Config.data_collection['image_crop_x2']]
            image = cv2.resize(image, 
                                (Config.neural_net['input_image_width'],
                                Config.neural_net['input_image_height']))
            image = image_process.process(image, bgr=True)
            images.append(image)
        for i in range(len(images)-int(Config.neural_net['lstm_timestep'])+1):
            images_timestep = []
            for j in range(Config.neural_net['lstm_timestep']):
                images_timestep.append(images[i+j])
            np_img = np.expand_dims(images_timestep, axis=0)
            del images_timestep
            
            layer_num = 3
            model = drive_run.net_model.model
            layer = model.layers[layer_num].output
            # for i in range(len(model.layers)):
                # print(model.layers[i].output)
            # print(model.layers[layer_num].output)
            model = Model(inputs=model.inputs, outputs=layer)
            feature_maps = model.predict(np_img)

            # plot all 64 maps in an 8x8 squares
            square = 8
            # print(feature_maps.shape[4])
            filter_num = int(feature_maps.shape[4])
            
            # feature_maps = cv2.resize(feature_maps, (feature_maps.shape[2], feature_maps.shape[2]))
            if filter_num >= square**2 :
                iter = filter_num // (square**2)
            else :
                iter = 1
            ix = 1
            for n in range(iter):
                plt_i = 1
                for _ in range(square):
                    for _ in range(square):
                        # specify subplot and turn of axis
                        # print(ix)
                        if ix < filter_num:
                            ax = plt.subplot(square, square, plt_i)
                            ax.set_xticks([])
                            ax.set_yticks([])
                            # print(ix)
                            # plot filter channel in grayscale
                            # feature_image = Image.fromarray(feature_maps[0, :, :, ix-1])
                            # feature_image = feature_image.resize((feature_maps.shape[2], feature_maps.shape[2]))
                            # feature_image = np.asarray(feature_image)
                            # plt.imshow(feature_image, cmap='gray')
                            # plt.imshow(feature_maps[0, -1, :, :, ix-1], cmap='gray')
                            plt.imshow(feature_maps[0, -1, :, :, ix-1], cmap='gray')
                            ix += 1
                            plt_i += 1
                        else:
                            break
            # show the figure
            # plt.show()
            
                saliency_file_path = image_folder_path + '/saliency' + '_' + model_path[-2:] + '/' + 'layer'+ str(layer_num) + '_' + str(n) + '_' + images_name[i][:-4] + '.png'
                # print(saliency_file_path)
                plt.savefig(saliency_file_path, dpi=150)
                plt.clf()
            cur_output = '{0}/{1}\r'.format(i, len(images_name))

            sys.stdout.write(cur_output)
            sys.stdout.flush()
            
        
    else :
        for i in range(len(images_name)):
            image_file_path = image_folder_path+'/'+images_name[i]
            image = cv2.imread(image_file_path)
            if Config.data_collection['crop'] is not True:
                image = image[Config.data_collection['image_crop_y1']:Config.data_collection['image_crop_y2'],
                            Config.data_collection['image_crop_x1']:Config.data_collection['image_crop_x2']]
            image = cv2.resize(image, 
                                (Config.neural_net['input_image_width'],
                                Config.neural_net['input_image_height']))
            image_raw = image_process.process(image, bgr=False)
            image = image_process.process(image, bgr=True)
            np_img = np.expand_dims(image, axis=0)
            measurement = drive_run.run((image, ))
            
            # fig = plt.figure()
            # plt.title('Prediction:\t' + str(measurement[0][0]) + '\nGroundTruth:\t' + str(steering_angles[i][0])
            #           + '\nError:\t' + str(abs(steering_angles[i][0] - measurement[0][0])))
            # layer_idx = utils.find_layer_idx(drive_run.net_model.model, 'conv_3')
            # filter_idx = utils.find_layer_idx(drive_run.net_model.model, 'maxpool_3')
            # filter_idx = None
            
            # # get filter weights
            # filters, biases = drive_run.net_model.model.layers[9].get_weights()

            # # normalize filter values to 0-1 so we can visualize them
            # f_min, f_max = filters.min(), filters.max()
            # filters = (filters - f_min) / (f_max - f_min)
            
            # n_filters, ix = 6, 1
            # for i in range(n_filters):
            #     # get the filter
            #     f = filters[:, :, :, i]
            #     # plot each channel separately
            #     for j in range(3):
            #         # specify subplot and turn of axis
            #         ax = plt.subplot(n_filters, 3, ix)
            #         ax.set_xticks([])
            #         ax.set_yticks([])
            #         # plot filter channel in grayscale
            #         plt.imshow(f[:, :, j], cmap='gray')
            #         ix += 1
            # # show the figure
            # plt.show()
            layer_num = 9
            model = drive_run.net_model.model
            layer = model.layers[layer_num].output
            print(layer)
            model = Model(inputs=model.inputs, outputs=layer)
            feature_maps = model.predict(np_img)

            # plot all 64 maps in an 8x8 squares
            square = 8
            # for _ in range(.)
            # print(feature_maps.shape[3])
            filter_num = int(feature_maps.shape[3])
            
            # feature_maps = cv2.resize(feature_maps, (feature_maps.shape[2], feature_maps.shape[2]))
            if filter_num >= square**2 :
                iter = filter_num // (square**2)
            else :
                iter = 1
            iter = 1
            ix = 1
            for n in range(iter):
                plt_i = 1
                for _ in range(square):
                    for _ in range(square):
                        # specify subplot and turn of axis
                        # print(ix)
                        if ix < filter_num:
                            ax = plt.subplot(square, square, plt_i)
                            ax.set_xticks([])
                            ax.set_yticks([])
                            # print(ix)
                            # plot filter channel in grayscale
                            # feature_image = Image.fromarray(feature_maps[0, :, :, ix-1])
                            # feature_image = feature_image.resize((feature_maps.shape[2], feature_maps.shape[2]))
                            # feature_image = np.asarray(feature_image)
                            # plt.imshow(feature_image, cmap='gray')
                            plt.imshow(feature_maps[0, :, :, ix-1], cmap='gray')
                            ix += 1
                            plt_i += 1
                        else:
                            break
            # show the figure
            # plt.show()
            
                saliency_file_path = image_folder_path + '/saliency' + '_' + model_path[-2:] + '/' + 'layer'+ str(layer_num) + '_' + images_name[i][:-4] + '_' + str(n) + '.png'
                # print(saliency_file_path)
                plt.savefig(saliency_file_path, dpi=150)
                plt.clf()
            cur_output = '{0}/{1}\r'.format(i, len(images_name))

            sys.stdout.write(cur_output)
            sys.stdout.flush()
            # # penultimate_layer_idx = utils.find_layer_idx(drive_run.net_model.model, 'conv2d_4')
            # fc_last = utils.find_layer_idx(drive_run.net_model.model, 'fc_str')
            # fc_last = utils.find_layer_idx(drive_run.net_model.model, 'fc_str')
            # heatmap = visualize_cam(drive_run.net_model.model, layer_idx, filter_indices=filter_idx, seed_input=image, backprop_modifier='guided')

            
            
            # ax1 = fig.add_subplot(2,1,1)
            # ax1.set_title('Prediction :' + str(format(measurement[0][0], ".9f")) 
            #         + '\nGroundTruth:' + str(format(steering_angles[i][0], ".9f"))
            #         + '\nError      :' + str(format(abs(steering_angles[i][0]-measurement[0][0]), ".9f"))
            #         )
            # plt.imshow(image)
            # plt.imshow(heatmap, cmap='jet', alpha=0.5)   
            # ax2 = fig.add_subplot(2,1,2) 
            # plt.subplot(2,1,2)
            # plt.imshow(image_raw)
                    
            # saliency_file_path = image_folder_path + '/saliency' + '_' + model_path[-2:] + '/' + images_name[i][:-4] + '_saliency.png'
            # plt.tight_layout()
            # # save fig    
            # plt.savefig(saliency_file_path, dpi=150)
            
            # cur_output = '{0}/{1}\r'.format(i, len(images_name))

            # sys.stdout.write(cur_output)
            # sys.stdout.flush()

# def show_layer_saliency(model_path, image_folder_path):
#     image_process = ImageProcess()
#     image_folder_name = None
#     if image_folder_path[-1] == '/':
#         image_folder_path = image_folder_path[:-1]
#     loc_slash = image_folder_path.rfind('/')
#     if loc_slash != -1: # there is '/' in the data path
#         image_folder_name = image_folder_path[loc_slash+1:] 
#     csv_path = image_folder_path+'/'+image_folder_name+'.csv'
    
#     dir_file_list = os.listdir(image_folder_path)
#     dir_img_list = [img for img in dir_file_list if img.endswith(".png") or img.endswith(".jpg")]
    
#     print(dir_img_list)
    
#     drive_run = DriveRun(model_path)
#     if os.path.isdir(image_folder_path + '/saliency'+ '_' + model_path[-2:]) is not True:
#         os.mkdir(image_folder_path + '/saliency'+ '_' + model_path[-2:])
    
#     num_conv_layer = 0
#     for i in range(len(drive_run.net_model.model.layers)):
#             if "Conv2D" in str(drive_run.net_model.model.layers[i]):
#                 num_conv_layer += 1
    
    
#     for j in range(len(dir_img_list)):
#         first_layer = True
#         fig = plt.figure(figsize=(3,10))
#         # fig, ax = plt.subplots(1, num_conv_layer-1)
        
#         num_layer = 1
#         for i in range(len(drive_run.net_model.model.layers)):
#             if "Conv2D" in str(drive_run.net_model.model.layers[i]):
#                 if first_layer is True:
#                     first_layer = False
#                 else:
#                     # print(drive_run.net_model.model.get_layer(index = i).name)
                    
#                     image_file_path = image_folder_path+'/'+dir_img_list[j]
#                     image = cv2.imread(image_file_path)
                    
#                     if Config.data_collection['crop'] is not True:
#                         image = image[Config.data_collection['image_crop_y1']:Config.data_collection['image_crop_y2'],
#                                     Config.data_collection['image_crop_x1']:Config.data_collection['image_crop_x2']]
#                     image = cv2.resize(image, 
#                                         (Config.neural_net['input_image_width'],
#                                         Config.neural_net['input_image_height']))
#                     image = image_process.process(image, bgr=False)
                    
#                     layer_idx = utils.find_layer_idx(drive_run.net_model.model, drive_run.net_model.model.get_layer(index = i).name)
#                     penultimate_layer_idx = utils.find_layer_idx(drive_run.net_model.model, drive_run.net_model.model.get_layer(index = i-1).name)
#                     fc_last = utils.find_layer_idx(drive_run.net_model.model, 'dense_5')
#                     # fc_last = utils.find_layer_idx(drive_run.net_model.model, 'fc_str')
#                     heatmap = visualize_cam(drive_run.net_model.model, layer_idx, 
#                                 filter_indices=fc_last, seed_input=image, backprop_modifier='guided', penultimate_layer_idx=penultimate_layer_idx)

#                     # ax[num_layer-1].imshow(image)
#                     # ax[num_layer-1].imshow(heatmap, cmap='jet', alpha=0.5)
#                     ax = fig.add_subplot(num_conv_layer-1,1, num_layer)
#                     # ax[num_layer-1].set_axis_off()
#                     plt.imshow(image)
#                     plt.imshow(heatmap, cmap='jet', alpha=0.5)
#                     num_layer += 1
#         plt.subplots_adjust(wspace=0, hspace=1)
#         plt.tight_layout()
#         saliency_file_path = image_folder_path + '/saliency' + '_' + model_path[-2:] + '/' + dir_img_list[j][:-4] + '_saliency_layers.png'
#         # save fig    
#         plt.savefig(saliency_file_path, dpi=150)
        
#         cur_output = '{0}/{1}\r'.format(j+1, len(dir_img_list))

#         sys.stdout.write(cur_output)
#         sys.stdout.flush()
#         del fig

# def main(model_path, image_file_path):
#     image_process = ImageProcess()

#     image = cv2.imread(image_file_path)

#     # if collected data is not cropped then crop here
#     # otherwise do not crop.
#     if Config.data_collection['crop'] is not True:
#         image = image[Config.data_collection['image_crop_y1']:Config.data_collection['image_crop_y2'],
#                       Config.data_collection['image_crop_x1']:Config.data_collection['image_crop_x2']]

#     image = cv2.resize(image, 
#                         (Config.neural_net['input_image_width'],
#                          Config.neural_net['input_image_height']))
#     image = image_process.process(image)

#     drive_run = DriveRun(model_path)
#     measurement = drive_run.run((image, ))

#     """ grad modifier doesn't work somehow
#     fig, axs = plt.subplots(1, 3)
#     fig.suptitle('Saliency Visualization' + str(measurement))
#     titles = ['left steering', 'right steering', 'maintain steering']
#     modifiers = [None, 'negate', 'small_values']

#     for i, modifier in enumerate(modifiers):
#         layer_idx = utils.find_layer_idx(drive_run.net_model.model, 'conv2d_last')
#         heatmap = visualize_cam(drive_run.net_model.model, layer_idx, 
#                     filter_indices=None, seed_input=image, backprop_modifier='guided', 
#                     grad_modifier=modifier)

#         axs[i].set(title = titles[i])
#         axs[i].imshow(image)
#         axs[i].imshow(heatmap, cmap='jet', alpha=0.3)
#     """
#     plt.figure()
#     #plt.title('Saliency Visualization' + str(measurement))
#     plt.title('Steering Angle Prediction: ' + str(measurement[0][0]))
#     layer_idx = utils.find_layer_idx(drive_run.net_model.model, 'conv2d_last')
#     heatmap = visualize_cam(drive_run.net_model.model, layer_idx, 
#                 filter_indices=None, seed_input=image, backprop_modifier='guided')

#     # plt.imshow(image)
#     # plt.imshow(heatmap, cmap='jet', alpha=0.5)

#     # file name
#     loc_slash = image_file_path.rfind('/')
#     if loc_slash != -1: # there is '/' in the data path
#         image_file_name = image_file_path[loc_slash+1:] 

#     saliency_file_path = model_path + '_' + image_file_name + '_saliency.png'
#     saliency_file_path_pdf = model_path + '_' + image_file_name + '_saliency.pdf'

#     plt.tight_layout()
#     # save fig    
#     plt.savefig(saliency_file_path, dpi=150)
#     plt.savefig(saliency_file_path_pdf, dpi=150)

#     print('Saved ' + saliency_file_path +' & .pdf')

#     # show the plot 
#     #plt.show()

###############################################################################
#       
if __name__ == '__main__':
    try:
        if (len(sys.argv) != 3):
            exit('Usage:\n$ python {} model_path, image_file_name'.format(sys.argv[0]))

        # main(sys.argv[1], sys.argv[2])
        
        images_cam(sys.argv[1], sys.argv[2])
        # images_saliency(sys.argv[1], sys.argv[2], Config.neural_net['lstm'])
        # show_layer_saliency(sys.argv[1], sys.argv[2])

    except KeyboardInterrupt:
        print ('\nShutdown requested. Exiting...')
