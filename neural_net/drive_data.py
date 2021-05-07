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

from config import Config

class DriveData:

    if Config.data_collection['brake'] is True:
        csv_header = ['image_fname', 'steering_angle', 'throttle', 'brake', 
                    'linux_time', 
                    'vel', 'vel_x', 'vel_y', 'vel_z',
                    'pos_x', 'pos_y', 'pos_z', 'accel_x', 'accel_y']
    else:
        csv_header = ['image_fname', 'steering_angle', 'throttle', 
                    'linux_time', 
                    'vel', 'vel_x', 'vel_y', 'vel_z',
                    'pos_x', 'pos_y', 'pos_z' ]

    def __init__(self, csv_fname):
        self.csv_fname = csv_fname
        self.df = None
        self.image_names = []
        self.measurements = []
        self.time_stamps = []
        self.velocities = []
        self.velocities_xyz = []
        self.positions_xyz = []

    def read(self, read = True, show_statistics = True, normalize = True):
        self.df = pd.read_csv(self.csv_fname, names=self.csv_header, index_col=False)
        #self.fname = fname

        ############################################
        # show statistics
        if (show_statistics):
            print('\n####### data statistics #########')
            print('Steering Command Statistics:')
            print(self.df['steering_angle'].describe())

            print('\nThrottle Command Statistics:')
            # Throttle Command Statistics
            print(self.df['throttle'].describe())

            if Config.data_collection['brake'] is True:
                print('\nBrake Command Statistics:')
                # Throttle Command Statistics
                print(self.df['brake'].describe())
                
            if Config.neural_net['num_outputs'] == 2:
                print('\nVelocity Command Statistics:')
                # Throttle Command Statistics
                print(self.df['vel'].describe())

        ############################################
        # normalize data
        # 'normalize' arg is for overriding 'normalize_data' config.
        if (Config.neural_net['normalize_data'] and normalize):
            print('\nnormalizing... wait for a moment')
            num_bins = 50
            fig, (ax1, ax2) = plt.subplots(1, 2)
            #fig.suptitle('Data Normalization')
            hist, bins = np.histogram(self.df['steering_angle'], (num_bins))
            center = (bins[:-1] + bins[1:])*0.5
            ax1.bar(center, hist, width=0.05)
            ax1.set(title = 'original')

            remove_list = []
            samples_per_bin = Config.neural_net['samples_per_bin']

            for j in range(num_bins):
                list_ = []
                for i in range(len(self.df['steering_angle'])):
                    if self.df.loc[i,'steering_angle'] >= bins[j] and self.df.loc[i,'steering_angle'] <= bins[j+1]:
                        list_.append(i)
                # random.shuffle(list_)
                list_ = list_[samples_per_bin:]
                remove_list.extend(list_)
            
            print('\r####### data normalization #########')
            print('removed:', len(remove_list))
            self.df.drop(self.df.index[remove_list], inplace = True)
            self.df.reset_index(inplace = True)
            self.df.drop(['index'], axis = 1, inplace = True)
            print('remaining:', len(self.df))
            
            hist, _ = np.histogram(self.df['steering_angle'], (num_bins))
            ax2.bar(center, hist, width=0.05)
            ax2.plot((np.min(self.df['steering_angle']), np.max(self.df['steering_angle'])), 
                        (samples_per_bin, samples_per_bin))  
            ax2.set(title = 'normalized')          

            plt.tight_layout()
            plt.savefig(self.get_data_path() + '_normalized.png', dpi=150)
            plt.savefig(self.get_data_path() + '_normalized.pdf', dpi=150)
            #plt.show()

        ############################################ 
        # read out
        if (read): 
            num_data = len(self.df)
            
            bar = ProgressBar()
            
            for i in bar(range(num_data)): # we don't have a title
                self.image_names.append(self.df.loc[i]['image_fname'])
                if Config.data_collection['brake'] is True:
                    self.measurements.append((float(self.df.loc[i]['steering_angle']),
                                            float(self.df.loc[i]['throttle']), 
                                            float(self.df.loc[i]['brake'])))
                else:
                    self.measurements.append((float(self.df.loc[i]['steering_angle']),
                                            float(self.df.loc[i]['throttle']), 
                                            0.0)) # dummy value for old data
                self.time_stamps.append(float(self.df.loc[i]['linux_time']))
                self.velocities.append(float(self.df.loc[i]['vel']))
                self.velocities_xyz.append((float(self.df.loc[i]['vel_x']), 
                                            float(self.df.loc[i]['vel_y']), 
                                            float(self.df.loc[i]['vel_z']),
                                            float(self.df.loc[i]['accel_x']), 
                                            float(self.df.loc[i]['accel_y'])))
                self.positions_xyz.append((float(self.df.loc[i]['pos_x']), 
                                            float(self.df.loc[i]['pos_y']), 
                                            float(self.df.loc[i]['pos_z'])))


    def get_data_path(self):
        loc_slash = self.csv_fname.rfind('/')
        
        if loc_slash != -1: # there is '/' in the data path
            data_path = self.csv_fname[:loc_slash] # get folder name
            return data_path
        else:
            exit('ERROR: csv file path must have a separator.')


###############################################################################
#  for testing DriveData class only
def main(data_path):
    import const

    if data_path[-1] == '/':
        data_path = data_path[:-1]

    loc_slash = data_path.rfind('/')
    if loc_slash != -1: # there is '/' in the data path
        model_name = data_path[loc_slash + 1:] # get folder name
        #model_name = model_name.strip('/')
    else:
        model_name = data_path
    csv_path = data_path + '/' + model_name + const.DATA_EXT   
    
    data = DriveData(csv_path)
    data.read(read = False)


###############################################################################
#       
if __name__ == '__main__':
    import sys

    try:
        if (len(sys.argv) != 2):
            exit('Usage:\n$ python {} data_path'.format(sys.argv[0]))

        main(sys.argv[1])

    except KeyboardInterrupt:
        print ('\nShutdown requested. Exiting...')
