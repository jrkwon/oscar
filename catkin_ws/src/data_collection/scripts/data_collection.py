#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 13:23:14 2017
History:
11/28/2020: modified for OSCAR 

@author: jaerock
"""

import rospy
import cv2
import os
import numpy as np
import datetime
import time
import sys
from geometry_msgs.msg import Twist
from sensor_msgs.msg import Image
from std_msgs.msg import String
from geometry_msgs.msg import Vector3Stamped
from nav_msgs.msg import Odometry
import math

import image_converter as ic
import const
from config import Config
import config 

##
# data will be saved in a location specified with rosparam path_to_e2e_data

if len(sys.argv) < 2:
    print('Usage: ')
    exit('$ rosrun data_collection data_collection.py your_data_id')


vehicle_steer = 0
vehicle_throttle = 0
vehicle_vel_x = vehicle_vel_y = vehicle_vel_z = 0
vehicle_vel = 0
vehicle_pos_x = vehicle_pos_y = vehicle_pos_z = 0


img_cvt = ic.ImageConverter()
config = Config.config

if config['vehicle_name'] == 'fusion':
    from fusion.msg import Control
elif config['vehicle_name'] == 'rover':
    from rover.msg import Control
else:
    exit(config['vehicle_name'] + 'not supported vehicle.')


name_datatime = str(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
#path = '../data/' + sys.argv[1] + '/' + name_datatime + '/'
path = rospy.get_param('path_to_e2e_data', './e2e_data') + '/' + sys.argv[1] + '/' + name_datatime + '/'
if os.path.exists(path):
    print('path exists. continuing...')
else:
    print('new folder created: ' + path)
    os.makedirs(path)

text = open(str(path) + name_datatime + const.DATA_EXT, "w+")

def vehicle_param_cb(value):
    global vehicle_throttle, vehicle_steer
    vehicle_throttle = value.throttle
    vehicle_steer = value.steer
    

def calc_vehicle_vel(x, y, z):
    return math.sqrt(vehicle_vel_x**2 + vehicle_vel_y**2 + vehicle_vel_z**2)


def vehicle_pos_vel_cb(value):
    global vehicle_pos_x, vehicle_pos_y, vehicle_pos_z
    global vehicle_vel_x, vehicle_vel_y, vehicle_vel_z, vehicle_vel

    vehicle_pos_x = value.pose.pose.position.x 
    vehicle_pos_y = value.pose.pose.position.y
    vehicle_pos_z = value.pose.pose.position.z

    vehicle_vel_x = value.twist.twist.linear.x 
    vehicle_vel_y = value.twist.twist.linear.y
    vehicle_vel_z = value.twist.twist.linear.z
    vehicle_vel = calc_vehicle_vel(vehicle_vel_x, vehicle_vel_y, vehicle_vel_z)


def recorder_cb(data):
    img = img_cvt.imgmsg_to_opencv(data)

    # no more cropping in data collection - new strategy    
    # # crop
    if config['crop'] is True: # this is for old datasets
        cropped = img[config['image_crop_y1']:config['image_crop_y2'],
                      config['image_crop_x1']:config['image_crop_x2']]

    unix_time = time.time()
    time_stamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
    file_full_path = str(path) + str(time_stamp) + const.IMAGE_EXT
    if config['crop'] is True:
        cv2.imwrite(file_full_path, cropped)
    else:
        cv2.imwrite(file_full_path, img)
    sys.stdout.write(file_full_path + '\r')
    line = "{}{},{},{},{},{},{},{},{},{},{},{}\r\n".format(time_stamp, const.IMAGE_EXT, 
                                                 vehicle_steer, 
                                                 vehicle_throttle,
                                                 unix_time,
                                                 vehicle_vel_x,
                                                 vehicle_vel_y,
                                                 vehicle_vel_z,
                                                 vehicle_vel,
                                                 vehicle_pos_x,
                                                 vehicle_pos_y,
                                                 vehicle_pos_z)
    text.write(line)                                                 
    """
    text.write(str(time_stamp) + const.IMAGE_EXT \
                + ',' + str(vehicle_steer) + ',' + str(vehicle_throttle) + \
                + ',' + str(unix_time) + \
                + ',' + str(vehicle_vel_x) + ',' + str(vehicle_vel_y) + ',' + str(vehicle_vel_z) + \
                + ',' + str(vehicle_vel) + \
                + ',' + str(vehicle_pos_x) + ',' + str(vehicle_pos_y) + ',' + str(vehicle_pos_z) + \
                "\r\n")
    """

def main():
    rospy.init_node('data_collection')
    rospy.Subscriber(config['vehicle_control_topic'], Control, vehicle_param_cb)
    rospy.Subscriber(config['camera_image_topic'], Image, recorder_cb)
    #rospy.Subscriber(config['velocity_topic'], Vector3Stamped, vehicle_velocity_cb)
    rospy.Subscriber(config['base_pose_topic'], Odometry, vehicle_pos_vel_cb)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        pass
    finally:
        print("\nBye...")    

if __name__ == '__main__':
    main()