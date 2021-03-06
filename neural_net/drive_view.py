#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 10 10:07:31 2021
History:
2/10/2021: modified for OSCAR 

@author: jaerock
"""

import cv2
import numpy as np
from progressbar import ProgressBar
from PIL import Image, ImageDraw, ImageFont

import const
from drive_data import DriveData
from config import Config
from image_process import ImageProcess
import os


class ImageSettings:
    def __init__(self, image_name, index_x):
        # local settings
        abs_path = os.environ['OSCAR_PATH'] + '/neural_net/'
        
        full_image_name = abs_path + image_name

        input_image_size = (Config.data_collection['image_width'], 
                            Config.data_collection['image_height'])

        margin_x = 50
        margin_y = 50
        spacer_x = 50

        # class settings
        self.image = Image.open(full_image_name)
        width, height = self.image.size
        self.image_pos = (margin_x + (width + spacer_x)*index_x, 
                         input_image_size[1] - height - margin_y)
        self.label_pos = (margin_x + (width + spacer_x)*index_x, 
                         self.image_pos[1] + height)


class DisplaySettings:
    def __init__(self):
        ##############################
        # information display settings
        self.info_pos = (10, 10)

        #########################
        # steering wheel settings
        self.label_wheel = ImageSettings('drive_view_img/steering_wheel_150x150.png', 0)
        self.infer_wheel = ImageSettings('drive_view_img/steering_wheel_green_150x150.png', 1)
        # logo
        self.logo = ImageSettings('drive_view_img/bimi_m_200x40.png', 2)
        
        ###############
        # font settings
        font_size = 20
        # Use fc-list to see installed fonts
        font_type = "FreeMonoBold.ttf"
        self.font = ImageFont.truetype(font_type, font_size)
        self.font_color = (255, 255, 255, 128) # white 50% transparent


###############################################################################
#
class DriveView:
    
    ###########################################################################
    # model_path = 'path_to_pretrained_model_name' excluding '.h5' or 'json'
    # data_path = 'path_to_drive_data'  e.g. ../data/2017-09-22-10-12-34-56'
    # target_path = path/to/save/view e.g. ../target/
    #    
    def __init__(self, model_path, data_path, target_path):
        # remove the last '/' in data and target path if exists
        if data_path[-1] == '/':
            data_path = data_path[:-1]
        if target_path[-1] == '/':
            target_path = target_path[:-1]

        loc_slash = data_path.rfind('/')
        if loc_slash != -1: # there is '/' in the data path
            data_name = data_path[loc_slash+1:] # get folder name
            #model_name = model_name.strip('/')
        else:
            data_name = data_path

        csv_path = data_path + '/' + data_name + const.DATA_EXT   
        
        self.data_name = data_name
        self.data_path = data_path
        self.target_path = target_path + '/' + data_name + '/'
        if os.path.isdir(target_path) is False:
            os.mkdir(target_path)
        if os.path.isdir(self.target_path) is False:
            os.mkdir(self.target_path)

        self.drive_data = DriveData(csv_path)
        self.drive_data.read(normalize=False)
        self.data_len = len(self.drive_data.image_names)

        self.net_model = None
        self.model_path = None
        if model_path is not None:
            from net_model import NetModel
            
            self.net_model = NetModel(model_path)
            self.net_model.load()
            self.model_path = model_path
        
        self.image_process = ImageProcess()

        self.display = DisplaySettings()

    def _print_info(self, i, draw, input_image, steering_angle, degree_angle):

        if self.net_model is not None and Config.neural_net['lstm'] is True:
            images = []
            lstm_time_step = 1


        ########################
        # inference included
        if self.net_model is not None:
            # convert PIL image to numpy array
            image = np.asarray(input_image)
            # don't forget OSCAR's default color space is BGR (cv2's default)
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # if collected data is not cropped then crop here
            # otherwise do not crop.
            if Config.data_collection['crop'] is not True:
                image = image[Config.data_collection['image_crop_y1']:Config.data_collection['image_crop_y2'],
                            Config.data_collection['image_crop_x1']:Config.data_collection['image_crop_x2']]

            image = cv2.resize(image, (Config.neural_net['input_image_width'],
                                    Config.neural_net['input_image_height']))

            image = self.image_process.process(image)

            ######################
            # infer using neural net
            if Config.neural_net['lstm'] is True:
                images.append(image)

                if lstm_time_step >= Config.neural_net['lstm_timestep']:
                    trans_image = np.array(images).reshape(-1, Config.neural_net['lstm_timestep'], 
                                                Config.neural_net['input_image_height'],
                                                Config.neural_net['input_image_width'],
                                                Config.neural_net['input_image_depth'])                    
                    predict = self.net_model.model.predict(trans_image)
                    pred_steering_angle = predict[0][0]
                    pred_steering_angle = pred_steering_angle / Config.neural_net['steering_angle_scale']
                    del images[0]
                lstm_time_step += 1
            else: # not lstm -- normal cnn
                npimg = np.expand_dims(image, axis=0)
                predict = self.net_model.model.predict(npimg)
                pred_steering_angle = predict[0][0]
                pred_steering_angle = pred_steering_angle / Config.neural_net['steering_angle_scale']

            #####################
            # display
            degree_angle = pred_steering_angle*Config.data_collection['steering_angle_max']
            rotated_img = self.display.infer_wheel.image.rotate(degree_angle)
            input_image.paste(rotated_img, self.display.infer_wheel.image_pos, rotated_img)

            draw.text(self.display.infer_wheel.label_pos, "Angle: {:.2f}".format(degree_angle), 
                        font=self.display.font, fill=self.display.font_color)


        if self.net_model is not None:
            diff = abs(pred_steering_angle - self.drive_data.measurements[i][0])
            if Config.data_collection['brake'] is True:
                draw.multiline_text(self.display.info_pos,
                            "Input:     {}\nThrottle:  {}\nBrake:     {}\nSteering:  {}\nPredicted: {}\nAbs Diff:  {}\nVelocity:  {:.2f}\nPosition:  (x:{:.2f}, y:{:.2f}, z:{:.2f})".format(
                                    self.drive_data.image_names[i], 
                                    self.drive_data.measurements[i][1],
                                    self.drive_data.measurements[i][2],
                                    # steering angle: -1 to 1 scale
                                    self.drive_data.measurements[i][0],
                                    pred_steering_angle,
                                    diff,
                                    self.drive_data.velocities[i], 
                                    self.drive_data.positions_xyz[i][0], 
                                    self.drive_data.positions_xyz[i][1], 
                                    self.drive_data.positions_xyz[i][2]), 
                                    font=self.display.font, fill=self.display.font_color)
            else:
                draw.multiline_text(self.display.info_pos,
                            "Input:     {}\nThrottle:  {}\nSteering:  {}\nPredicted: {}\nAbs Diff:  {}\nVelocity:  {:.2f}\nPosition:  (x:{:.2f}, y:{:.2f}, z:{:.2f})".format(
                                    self.drive_data.image_names[i], 
                                    self.drive_data.measurements[i][1],
                                    # steering angle: -1 to 1 scale
                                    self.drive_data.measurements[i][0],
                                    pred_steering_angle,
                                    diff,
                                    self.drive_data.velocities[i], 
                                    self.drive_data.positions_xyz[i][0], 
                                    self.drive_data.positions_xyz[i][1], 
                                    self.drive_data.positions_xyz[i][2]), 
                                    font=self.display.font, fill=self.display.font_color)

            loc_dot = self.drive_data.image_names[i].rfind('.')
            target_img_name = "{}_{:.2f}_{:.2f}{}".format(self.drive_data.image_names[i][:loc_dot], 
                                                        pred_steering_angle, degree_angle, const.IMAGE_EXT)
        else:
            if Config.data_collection['brake'] is True:
                draw.multiline_text(self.display.info_pos,
                            "Input:     {}\nThrottle:  {}\nBrake:     {}\nSteering:  {}\nVelocity: {:.2f}\nPosition: (x:{:.2f}, y:{:.2f}, z:{:.2f})".format(
                                    self.drive_data.image_names[i], 
                                    self.drive_data.measurements[i][1],
                                    self.drive_data.measurements[i][2],
                                    # steering angle: -1 to 1 scale
                                    self.drive_data.measurements[i][0],
                                    self.drive_data.velocities[i], 
                                    self.drive_data.positions_xyz[i][0], 
                                    self.drive_data.positions_xyz[i][1], 
                                    self.drive_data.positions_xyz[i][2]), 
                                    font=self.display.font, fill=self.display.font_color)
            else:
                draw.multiline_text(self.display.info_pos,
                            "Input:     {}\nThrottle:  {}\nSteering:  {}\nVelocity: {:.2f}\nPosition: (x:{:.2f}, y:{:.2f}, z:{:.2f})".format(
                                    self.drive_data.image_names[i], 
                                    self.drive_data.measurements[i][1],
                                    # steering angle: -1 to 1 scale
                                    self.drive_data.measurements[i][0],
                                    self.drive_data.velocities[i], 
                                    self.drive_data.positions_xyz[i][0], 
                                    self.drive_data.positions_xyz[i][1], 
                                    self.drive_data.positions_xyz[i][2]), 
                                    font=self.display.font, fill=self.display.font_color)
            
            loc_dot = self.drive_data.image_names[i].rfind('.')
            target_img_name = "{}_{:.2f}_{:.2f}{}".format(self.drive_data.image_names[i][:loc_dot], 
                                                        steering_angle, degree_angle, const.IMAGE_EXT)
        # save it
        input_image.save(self.target_path + target_img_name)

    ###########################################################################
    #
    def run(self):
        
        bar = ProgressBar()

        ############################
        # steering angle raw value:
        # -1 to 1 (0 --> 1: left, 0 --> -1: right)
        for i in bar(range(self.data_len)):
            abs_path_image = self.data_path + '/' + self.drive_data.image_names[i]
            input_image = Image.open(abs_path_image)
            steering_angle = self.drive_data.measurements[i][0] # -1 to 1 scale
            degree_angle = steering_angle*Config.data_collection['steering_angle_max']
            rotated_img = self.display.label_wheel.image.rotate(degree_angle)
            input_image.paste(rotated_img, self.display.label_wheel.image_pos, rotated_img)    

            # logo
            input_image.paste(self.display.logo.image, self.display.logo.image_pos, self.display.logo.image)

            draw = ImageDraw.Draw(input_image)
            draw.text(self.display.label_wheel.label_pos, "Angle: {:.2f}".format(degree_angle), 
                    font=self.display.font, fill=self.display.font_color)

            self._print_info(i, draw, input_image, steering_angle, degree_angle)



###############################################################################
#  for testing DriveView      
def main(weight_name, data_folder_name, target_folder_name):
    drive_view = DriveView(weight_name, data_folder_name, target_folder_name) 
    drive_view.run() # data folder path to test
       

###############################################################################
#       
if __name__ == '__main__':
    import sys
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
