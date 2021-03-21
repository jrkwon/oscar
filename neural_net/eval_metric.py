#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sun 21 Mar 19∶20∶27: 2021
History:

@author: Donghyun Kim
"""

import cv2
import numpy as np
from progressbar import ProgressBar
import matplotlib.pyplot as plt

import const
from config import Config
from net_model import NetModel
from drive_data import DriveData
from image_process import ImageProcess

###############################################################################
#
class Evalmetric:
    
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

    ###########################################################################
    #
    def _calc_groundtruth(self):
        # 1. 전체 지도의 정보를 받아옴 ( 타일 형태, 타일 길이 )
        # 2. 초기 시작점 (0,0)을 기준으로 타일별 위치를 구함
        # 3. 맵의 중앙선을 계산
        
        pass
    # def run(self):
        
        



from eval_metric import Evalmetric


###############################################################################
#       
def main(weight_name, data_folder_name):
    eval_metric = Evalmetric(weight_name, data_folder_name) 
    # eval_metric.run() # data folder path to test
       

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
