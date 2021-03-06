#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 15 22:07:31 2019
History:
11/28/2020: modified for OSCAR 

@author: jaerock
"""

import cv2
import numpy as np
#import keras
#import sklearn
#import resnet
from progressbar import ProgressBar
import matplotlib.pyplot as plt

import const
from net_model import NetModel
from drive_data import DriveData
from config import Config
from image_process import ImageProcess

###############################################################################
#
class DriveLog:
    
    ###########################################################################
    # model_path = 'path_to_pretrained_model_name' excluding '.h5' or 'json'
    # data_path = 'path_to_drive_data'  e.g. ../data/2017-09-22-10-12-34-56'
       
    def __init__(self, model_path, data_path):
        if data_path[-1] == '/':
            data_path = data_path[:-1]

        loc_slash = data_path.rfind('/')
        if loc_slash != -1: # there is '/' in the data path
            model_name = data_path[loc_slash+1:] # get folder name
            #model_name = model_name.strip('/')
        else:
            model_name = data_path

        csv_path = data_path + '/' + model_name + const.DATA_EXT   
        
        self.data_path = data_path
        self.data = DriveData(csv_path)
        
        self.test_generator = None
        
        self.num_test_samples = 0        
        #self.config = Config()
        
        self.net_model = NetModel(model_path)
        self.net_model.load()
        self.model_path = model_path
        
        self.image_process = ImageProcess()

        self.measurements = []
        self.predictions = []
        self.differences = []
        self.squared_differences = []

    ###########################################################################
    #
    def _prepare_data(self):
        
        self.data.read(normalize = False)
    
        self.test_data = list(zip(self.data.image_names, self.data.velocities, self.data.measurements))
        self.num_test_samples = len(self.test_data)
        
        print('Test samples: {0}'.format(self.num_test_samples))

    
   ###########################################################################
    #
    def _savefigs(self, plt, filename):
        plt.savefig(filename + '.png', dpi=150)
        plt.savefig(filename + '.pdf', dpi=150)
        print('Saved ' + filename + '.png & .pdf.')


    ###########################################################################
    #
    def _plot_results(self):
        plt.figure()
        # Plot a histogram of the prediction errors
        num_bins = 25
        hist, bins = np.histogram(self.differences, num_bins)
        center = (bins[:-1]+ bins[1:]) * 0.5
        plt.bar(center, hist, width=0.05)
        #plt.title('Historgram of Predicted Errors')
        plt.xlabel('Steering Angle')
        plt.ylabel('Number of Predictions')
        plt.xlim(-1.0, 1.0)
        plt.plot(np.min(self.differences), np.max(self.differences))
        plt.tight_layout()
        self._savefigs(plt, self.data_path + '_err_hist')

        plt.figure()
        # Plot a Scatter Plot of the Error
        plt.scatter(self.measurements, self.predictions)
        #plt.title('Scatter Plot of Errors')
        plt.xlabel('True Values')
        plt.ylabel('Predictions')
        plt.axis('equal')
        plt.axis('square')
        plt.xlim([-1.0, 1.0])
        plt.ylim([-1.0, 1.0])
        plt.plot([-1.0, 1.0], [-1.0, 1.0], color='k', linestyle='-', linewidth=.1)
        plt.tight_layout()
        self._savefigs(plt, self.data_path + '_scatter')

        plt.figure()
        # Plot a Side-By-Side Comparison
        plt.plot(self.measurements)
        plt.plot(self.predictions)
        mean = sum(self.differences)/len(self.differences)
        variance = sum([((x - mean) ** 2) for x in self.differences]) / len(self.differences) 
        std = variance ** 0.5
        plt.title('MAE: {0:.3f}, STDEV: {1:.3f}'.format(mean, std))
        #plt.title('Ground Truth vs. Prediction')
        plt.ylim([-1.0, 1.0])
        plt.xlabel('Time Step')
        plt.ylabel('Steering Angle')
        plt.legend(['ground truth', 'prediction'], loc='upper right')
        plt.tight_layout()
        self._savefigs(plt, self.data_path + '_comparison')

        # show all figures
        #plt.show()


   ###########################################################################
    #
    def run(self):
        
        self._prepare_data()
        #fname = self.data_path + const.LOG_EXT
        fname = self.data_path + const.LOG_EXT # use model name to save log
        
        file = open(fname, 'w')

        #print('image_name', 'label', 'predict', 'abs_error')
        bar = ProgressBar()

        file.write('image_name, label_steering_angle, pred_steering_angle, abs_error, squared_error\n')

        if Config.neural_net['lstm'] is True:
            images = []
            #images_names = []
            cnt = 1

            for image_name, velocity, measurement in bar(self.test_data):   
                image_fname = self.data_path + '/' + image_name
                image = cv2.imread(image_fname)

                # if collected data is not cropped then crop here
                # otherwise do not crop.
                if Config.data_collection['crop'] is not True:
                    image = image[Config.data_collection['image_crop_y1']:Config.data_collection['image_crop_y2'],
                                Config.data_collection['image_crop_x1']:Config.data_collection['image_crop_x2']]

                image = cv2.resize(image, (Config.neural_net['input_image_width'],
                                        Config.neural_net['input_image_height']))
                image = self.image_process.process(image)

                images.append(image)
                #images_names.append(image_name)
                
                if cnt >= Config.neural_net['lstm_timestep']:
                    trans_image = np.array(images).reshape(-1, Config.neural_net['lstm_timestep'], 
                                                Config.neural_net['input_image_height'],
                                                Config.neural_net['input_image_width'],
                                                Config.neural_net['input_image_depth'])                    

                    predict = self.net_model.model.predict(trans_image)
                    pred_steering_angle = predict[0][0]
                    pred_steering_angle = pred_steering_angle / Config.neural_net['steering_angle_scale']
                
                    if Config.neural_net['num_outputs'] == 2:
                        pred_throttle = predict[0][1]
                    
                    label_steering_angle = measurement[0] # labeled steering angle
                    self.measurements.append(label_steering_angle)
                    self.predictions.append(pred_steering_angle)
                    diff = abs(label_steering_angle - pred_steering_angle)
                    self.differences.append(diff)
                    self.squared_differences.append(diff*2)
                    log = image_name+','+str(label_steering_angle)+','+str(pred_steering_angle)\
                                    +','+str(diff)\
                                    +','+str(diff**2)

                    file.write(log+'\n')
                    # delete the 1st element
                    del images[0]
                cnt += 1
        else:
            for image_name, velocity, measurement in bar(self.test_data):   
                image_fname = self.data_path + '/' + image_name
                image = cv2.imread(image_fname)

                # if collected data is not cropped then crop here
                # otherwise do not crop.
                if Config.data_collection['crop'] is not True:
                    image = image[Config.data_collection['image_crop_y1']:Config.data_collection['image_crop_y2'],
                                Config.data_collection['image_crop_x1']:Config.data_collection['image_crop_x2']]

                image = cv2.resize(image, (Config.neural_net['input_image_width'],
                                        Config.neural_net['input_image_height']))
                image = self.image_process.process(image)
                
                npimg = np.expand_dims(image, axis=0)
                predict = self.net_model.model.predict(npimg)
                pred_steering_angle = predict[0][0]
                pred_steering_angle = pred_steering_angle / Config.neural_net['steering_angle_scale']
                
                if Config.neural_net['num_outputs'] == 2:
                    pred_throttle = predict[0][1]

                label_steering_angle = measurement[0]
                self.measurements.append(label_steering_angle)
                self.predictions.append(pred_steering_angle)
                diff = abs(label_steering_angle - pred_steering_angle)
                self.differences.append(diff)
                self.squared_differences.append(diff**2)
                #print(image_name, measurement[0], predict,\ 
                #                  abs(measurement[0]-predict))
                log = image_name+','+str(label_steering_angle) + ',' + str(pred_steering_angle)\
                                +','+str(diff)\
                                +','+str(diff**2)

                file.write(log+'\n')
        
        file.close()
        print('Saved ' + fname + '.')

        self._plot_results()



from drive_log import DriveLog


###############################################################################
#       
def main(weight_name, data_folder_name):
    drive_log = DriveLog(weight_name, data_folder_name) 
    drive_log.run() # data folder path to test
       

###############################################################################
#       
if __name__ == '__main__':
    import sys

    try:
        if (len(sys.argv) != 3):
            exit('Usage:\n$ python {} weight_name data_folder_name'.format(sys.argv[0]))
        
        main(sys.argv[1], sys.argv[2])

    except KeyboardInterrupt:
        print ('\nShutdown requested. Exiting...')
