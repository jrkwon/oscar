#!/usr/bin/env python
# -*- coding: utf-8 -*-

import rospy
import cv2
import os
import numpy as np
import datetime
import time
import sys
from geometry_msgs.msg import Point, PoseStamped, Quaternion, Vector3, Twist, TwistStamped
from mavros_msgs.msg import AttitudeTarget
from mavros_msgs.msg import *
from mavros_msgs.srv import *
from std_msgs.msg import Header, String
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
import math
import image_converter as ic
import const
 
from rover.msg import Control

from config import Config
import config
config = Config.data_collection

class DataCollection():
    def __init__(self):
        self.steering = 0
        self.throttle = 0
        self.brake = 0

        self.vel_x = self.vel_y = self.vel_z = 0
        self.vel = 0
        self.pos_x = self.pos_y = self.pos_z = 0

        self.img_cvt = ic.ImageConverter()

        ##
        # data will be saved in a location specified with rosparam path_to_e2e_data

        # create csv data file
        name_datatime = str(datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S"))
        #path = '../data/' + sys.argv[1] + '/' + name_datatime + '/'
        path = rospy.get_param('path_to_e2e_data', 
                        './e2e_data') + '/' + sys.argv[1] + '/' + name_datatime + '/'
        if os.path.exists(path):
            print('path exists. continuing...')
        else:
            print('new folder created: ' + path)
            os.makedirs(path)

        self.text = open(str(path) + name_datatime + const.DATA_EXT, "w+")
        self.path = path


    def calc_velocity(self, x, y, z):
        return math.sqrt(x**2 + y**2 + z**2)


    def steering_throttle_cb(self, value):
        self.throttle = value.throttle
        self.steering = value.steer
        self.brake = value.brake


    def pos_vel_cb(self, value):

        self.pos_x = value.pose.pose.position.x 
        self.pos_y = value.pose.pose.position.y
        self.pos_z = value.pose.pose.position.z

        self.vel_x = value.twist.twist.linear.x 
        self.vel_y = value.twist.twist.linear.y
        self.vel_z = value.twist.twist.linear.z
        self.vel = self.calc_velocity(self.vel_x, self.vel_y, self.vel_z)


    def recorder_cb(self, data):
        img = self.img_cvt.imgmsg_to_opencv(data)

        # no more cropping in data collection - new strategy    
        # # crop

        if config['crop'] is True: # this is for old datasets
            cropped = img[config['image_crop_y1']:config['image_crop_y2'],
                          config['image_crop_x1']:config['image_crop_x2']]

        unix_time = time.time()
        time_stamp = datetime.datetime.now().strftime("%Y-%m-%d-%H-%M-%S-%f")
        file_full_path = str(self.path) + str(time_stamp) + const.IMAGE_EXT
        if config['crop'] is True:
            cv2.imwrite(file_full_path, cropped)
        else:
            cv2.imwrite(file_full_path, img)
        sys.stdout.write(file_full_path + '\r')
        if config['version'] >= 0.92:
            line = "{}{},{},{},{},{},{},{},{},{},{},{},{}\r\n".format(time_stamp, const.IMAGE_EXT, 
                                                        self.steering, 
                                                        self.throttle,
                                                        self.brake,
                                                        unix_time,
                                                        self.vel,
                                                        self.vel_x,
                                                        self.vel_y,
                                                        self.vel_z,
                                                        self.pos_x,
                                                        self.pos_y,
                                                        self.pos_z)
        else:
            line = "{}{},{},{},{},{},{},{},{},{},{},{}\r\n".format(time_stamp, const.IMAGE_EXT, 
                                                        self.steering, 
                                                        self.throttle,
                                                        unix_time,
                                                        self.vel,
                                                        self.vel_x,
                                                        self.vel_y,
                                                        self.vel_z,
                                                        self.pos_x,
                                                        self.pos_y,
                                                        self.pos_z)
        self.text.write(line)                                                 


def main():
    dc = DataCollection()

    rospy.init_node('data_collection')
    rospy.Subscriber('/rover', Control, dc.steering_throttle_cb)
    #rospy.Subscriber('/mavros/global_position/local', Odometry, dc.pos_vel_cb)
    rospy.Subscriber('/mavros/odometry/in', Odometry, dc.pos_vel_cb)
    rospy.Subscriber('/camera/color/image_raw', Image, dc.recorder_cb)

    try:
        rospy.spin()
    except KeyboardInterrupt:
        pass
    finally:
        print("\nBye...")    


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print('Usage: ')
        exit('$ rosrun data_collection data_collection1.py your_data_id')

    main()
