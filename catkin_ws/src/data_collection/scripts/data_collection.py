#!/usr/bin/env python3
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
vehicle_vel = 0

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

def vehicle_param(value):
    global vehicle_vel, vehicle_steer
    vehicle_vel = value.throttle
    vehicle_steer = value.steer
    #return (vehicle_vel, vehicle_steer)

def recorder(data):
    img = img_cvt.imgmsg_to_opencv(data)

    # crop
    cropped = img[config['image_crop_y1']:config['image_crop_y2'],
                  config['image_crop_x1']:config['image_crop_x2']]

    time_stamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
    file_full_path = str(path) + str(time_stamp) + const.IMAGE_EXT
    cv2.imwrite(file_full_path, cropped)
    sys.stdout.write(file_full_path + '\r')
    text.write(str(time_stamp) + const.IMAGE_EXT + ',' + str(vehicle_steer) + ',' + str(vehicle_vel) + "\r\n")
    
def main():
    rospy.init_node('data_collection')
    rospy.Subscriber(config['vehicle_control_topic'], Control, vehicle_param)
    rospy.Subscriber(config['camera_image_topic'], Image, recorder)
    try:
        rospy.spin()
    except KeyboardInterrupt:
        pass
    finally:
        print("\nBye...")    

if __name__ == '__main__':
    main()