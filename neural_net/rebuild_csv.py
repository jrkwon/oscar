#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 12:23:14 2019
History:
11/28/2020: modified for OSCAR 

@author: jaerock
"""


import sys
import os
from progressbar import ProgressBar

import const
from drive_data import DriveData
from config import Config

###############################################################################
#
def build_csv(data_path):
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

    csv_backup_name = data_path + csv_file + '.bak'
    os.rename(data_path + csv_file, csv_backup_name)
    print('rename ' + data_path + csv_file + ' to ' + csv_backup_name)

    data = DriveData(csv_backup_name)
    data.read(normalize = False)

    new_csv = []

    # check image exists
    bar = ProgressBar()
    for i in bar(range(len(data.df))):
        if os.path.exists(data_path + data.image_names[i]):
            if Config.data_collection['brake'] is True:
                new_csv.append(data.image_names[i] + ','
                            + str(data.measurements[i][0]) + ','
                            + str(data.measurements[i][1]) + ','
                            + str(data.measurements[i][2]) + ',' # brake
                            + str(data.time_stamps[i]) + ','
                            + str(data.velocities[i]) + ','
                            + str(data.velocities_xyz[i][0]) + ','
                            + str(data.velocities_xyz[i][1]) + ','
                            + str(data.velocities_xyz[i][2]) + ','
                            + str(data.positions_xyz[i][0]) + ','
                            + str(data.positions_xyz[i][1]) + ','
                            + str(data.positions_xyz[i][2]) + '\n')
            else:
                new_csv.append(data.image_names[i] + ','
                            + str(data.measurements[i][0]) + ','
                            + str(data.measurements[i][1]) + ','
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
        print('Usage: \n$ python rebuild_csv data_folder_name')
        return

    build_csv(sys.argv[1])


###############################################################################
#
if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nShutdown requested. Exiting...')
