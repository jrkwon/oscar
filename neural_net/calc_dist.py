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
import math
import matplotlib.pyplot as plt


###############################################################################
#
def calc_dist(data_path):
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

    data = DriveData(data_path + csv_file)
    data.read(normalize = False)

    bar = ProgressBar()
    rows = len(data.df)
    assert rows > 0, "data len is 0"

    x = []
    y = []
    dist = 0
    for i in bar(range(rows - 1)):
        x_diff = data.positions_xyz[i + 1][0] - data.positions_xyz[i][0]
        y_diff = data.positions_xyz[i + 1][1] - data.positions_xyz[i][1]
        z_diff = data.positions_xyz[i + 1][2] - data.positions_xyz[i][2]

        dist += math.sqrt(x_diff**2 + y_diff**2 + z_diff**2)
        x.append(data.positions_xyz[i][0])
        y.append(data.positions_xyz[i][1])
        """
                    new_csv.append(data.image_names[i] + ','
                           + str(data.measurements[i][0]) + ','
                           + str(data.measurements[i][1]) + ','
                           + str(data.time_stamps[i]) + ','
                           + str(data.velocities[i]) + ','
                           + str(data.velocities_xyz[i][0]) + ','
                           + str(data.velocities_xyz[i][1]) + ','
                           + str(data.velocities_xyz[i][2]) + ','
        """
    x.append(data.positions_xyz[-1][0])
    y.append(data.positions_xyz[-1][1])
    
    # plot travel trajectory
    plt.figure()
    # Plot a Scatter Plot 
    plt.scatter(x, y, s=1, marker='o')
    #plt.title('Travel Trajectory')
    plt.xlabel('x')
    plt.ylabel('y')
    #plt.axis('equal')
    #plt.axis('square')
    plt.title('Travel Distance: {0:.2f} meters'.format(dist))
    #plt.xlim([-1.0, 1.0])
    #plt.ylim([-1.0, 1.0])
    #plt.plot([-1.0, 1.0], [-1.0, 1.0], color='k', linestyle='-', linewidth=.1)
    plt.tight_layout()
    plt.savefig(data_path[:-1] + '_travel.png', dpi=150)
    plt.savefig(data_path[:-1] + '_travel.pdf', dpi=150)
    print('Saved ' + data_path[:-1] + '_travel.png & pdf')

    plt.figure()
    plt.plot(data.velocities)
    plt.xlabel('time step')
    plt.ylabel('vel (meters/h)')
    plt.tight_layout()
    plt.savefig(data_path[:-1] + '_vel.png', dpi=150)
    plt.savefig(data_path[:-1] + '_vel.pdf', dpi=150)
    print('Saved ' + data_path[:-1] + '_vel.png & pdf')

    #plt.show()

    return dist

###############################################################################
#
def main():
    if (len(sys.argv) != 2):
        print('Usage: \n$ python ' + sys.arg[0] + ' data_folder_name')
        return

    print("Total travel distance (meters): {0:.2f}".format(calc_dist(sys.argv[1])))


###############################################################################
#
if __name__ == '__main__':
    try:
        main()
    except KeyboardInterrupt:
        print('\nShutdown requested. Exiting...')
