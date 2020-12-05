#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 13:23:14 2017
History:
11/28/2020: modified for OSCAR 

@author: jaerock
"""


import pandas as pd
from progressbar import ProgressBar
import matplotlib.pyplot as plt
import numpy as np
import random

class DriveData:
    
    csv_header = ['image_fname', 'steering_angle', 'throttle']
    
    def __init__(self, csv_fname):
        self.csv_fname = csv_fname
        self.df = None
        self.image_names = []
        self.measurements = []
    
    def read(self, normalize_data = False):
        self.df = pd.read_csv(self.csv_fname, names=self.csv_header, index_col=False)
        #self.fname = fname

        ############################################
        # normalize data

        if (normalize_data):
            num_bins = 50
            _, (ax1, ax2) = plt.subplots(1, 2)
            hist, bins = np.histogram(self.df['steering_angle'], num_bins)
            center = (bins[:-1] + bins[1:])*0.5
            ax1.bar(center, hist)

            remove_list = []
            samples_per_bin = 200

            for j in range(num_bins):
                list_ = []
                for i in range(len(self.df['steering_angle'])):
                    if self.df.loc[i,'steering_angle'] >= bins[j] and self.df.loc[i,'steering_angle'] <= bins[j+1]:
                        list_.append(i)
                random.shuffle(list_)
                list_ = list_[samples_per_bin:]
                remove_list.extend(list_)
            print('####### data normalization #########')
            print('removed:', len(remove_list))
            self.df.drop(self.df.index[remove_list], inplace = True)
            self.df.reset_index(inplace = True)
            self.df.drop(['index'], axis = 1, inplace = True)
            print('remaining:', len(self.df))
            
            hist, _ = np.histogram(self.df['steering_angle'], (num_bins))
            ax2.bar(center, hist, width=0.05)
            ax2.plot((np.min(self.df['steering_angle']), np.max(self.df['steering_angle'])), 
                        (samples_per_bin, samples_per_bin))            

            plt.show()

        ############################################ 
        # read out
        num_data = len(self.df)
        
        bar = ProgressBar()
        
        for i in bar(range(num_data)): # we don't have a title
            self.image_names.append(self.df.loc[i]['image_fname'])
            self.measurements.append((float(self.df.loc[i]['steering_angle']),
                                        float(self.df.loc[i]['throttle'])))

