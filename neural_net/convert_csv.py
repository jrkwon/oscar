#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Feb 17 12:23:14 2021
History:
02/17/2021: modified for OSCAR 
@author: jaerock
"""

"""
csv converter: 
    csv (before 0.92) -->  new_csv (adding dummy brake)
"""

import sys
import os
import pandas as pd
from progressbar import ProgressBar
import const

class AddDummyBrake:

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

    def read(self, read = True, show_statistics = True):
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

        ############################################ 
        # read out
        if (read): 
            num_data = len(self.df)
            
            bar = ProgressBar()
            
            for i in bar(range(num_data)): # we don't have a title
                self.image_names.append(self.df.loc[i]['image_fname'])
                self.measurements.append((float(self.df.loc[i]['steering_angle']),
                                        float(self.df.loc[i]['throttle']), 
                                        0.0)) # dummy value for old data
                self.time_stamps.append(float(self.df.loc[i]['linux_time']))
                self.velocities.append(float(self.df.loc[i]['vel']))
                self.velocities_xyz.append((float(self.df.loc[i]['vel_x']), 
                                            float(self.df.loc[i]['vel_y']), 
                                            float(self.df.loc[i]['vel_z'])))
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
#
def convert_csv(data_path):
    # add '/' at the end of data_path if user doesn't specify
    if data_path[-1] != '/':
        data_path = data_path + '/'

    # find the second '/' from the end to get the folder name
    loc_dir_delim = data_path[:-1].rfind('/')
    if (loc_dir_delim != -1):
        folder_name = data_path[loc_dir_delim+1:-1]
    else:
        folder_name = data_path[:-1]

    csv_file = folder_name + const.DATA_EXT

    csv_backup_name = data_path + csv_file + '.old'
    os.rename(data_path + csv_file, csv_backup_name)

    data = AddDummyBrake(csv_backup_name)
    data.read()

    new_csv = []

    # check image exists
    bar = ProgressBar()
    for i in bar(range(len(data.df))):
        new_csv.append(data.image_names[i] + ','
                        + str(data.measurements[i][0]) + ',' # steering
                        + str(data.measurements[i][1]) + ',' # throttle
                        + str(data.measurements[i][2]) + ',' # brake
                        + str(data.time_stamps[i]) + ','
                        + str(data.velocities[i]) + ','
                        + str(data.velocities_xyz[i][0]) + ','
                        + str(data.velocities_xyz[i][1]) + ','
                        + str(data.velocities_xyz[i][2]) + ','
                        + str(data.positions_xyz[i][0]) + ','
                        + str(data.positions_xyz[i][1]) + ','
                        + str(data.positions_xyz[i][2]) + '\n')

    # write a new csv
    new_csv_fh = open(data_path + csv_file, 'w')
    for i in range(len(new_csv)):
        new_csv_fh.write(new_csv[i])
    new_csv_fh.close()


###############################################################################
#
def main():
    if (len(sys.argv) != 2):
        print('Usage: \n$ python {} data_folder_name'.format(sys.argv[0]))
        return

    convert_csv(sys.argv[1])


###############################################################################
#
if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nShutdown requested. Exiting...')
