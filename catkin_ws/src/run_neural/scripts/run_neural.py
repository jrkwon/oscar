#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sat Sep 23 13:23:14 2017
History:
11/28/2020: modified for OSCAR 

@author: jaerock
"""

import threading 
import cv2
import time
import rospy
import numpy as np
from fusion.msg import Control
from std_msgs.msg import Int32
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
import math

import sys
import os

#sys.path.append('../neural_net/')
#os.chdir('../neural_net/')

import const
from image_converter import ImageConverter
from drive_run import DriveRun
from config import Config
from image_process import ImageProcess

config = Config.config
if config['vehicle_name'] == 'fusion':
    from fusion.msg import Control
elif config['vehicle_name'] == 'rover':
    from rover.msg import Control
else:
    exit(config['vehicle_name'] + 'not supported vehicle.')

velocity = 0


class NeuralControl:
    def __init__(self, weight_file_name):
        rospy.init_node('run_neural')
        self.ic = ImageConverter()
        self.image_process = ImageProcess()
        self.rate = rospy.Rate(30)
        self.drive= DriveRun(weight_file_name)
        rospy.Subscriber(config['camera_image_topic'], Image, self._controller_cb)
        self.image = None
        self.image_processed = False
        #self.config = Config()
        self.braking = False

    def _controller_cb(self, image): 
        img = self.ic.imgmsg_to_opencv(image)
        cropped = img[config['image_crop_y1']:config['image_crop_y2'],
                      config['image_crop_x1']:config['image_crop_x2']]
                      
        img = cv2.resize(cropped, (config['input_image_width'],
                                   config['input_image_height']))
                                  
        self.image = self.image_process.process(img)

        ## this is for CNN-LSTM net models
        if config['lstm'] is True:
            self.image = np.array(self.image).reshape(1, 
                                 config['input_image_height'],
                                 config['input_image_width'],
                                 config['input_image_depth'])
        self.image_processed = True
        
    def timer_cb(self):
        self.braking = False
      
def pos_vel_cb(value):
    global velocity

    vel_x = value.twist.twist.linear.x 
    vel_y = value.twist.twist.linear.y
    vel_z = value.twist.twist.linear.z
    
    velocity = math.sqrt(vel_x**2 + vel_y**2 + vel_z**2)


        
def main(weight_file_name):

    # ready for neural network
    neural_control = NeuralControl(weight_file_name)
    
    rospy.Subscriber(config['base_pose_topic'], Odometry, pos_vel_cb)
    # ready for /bolt topic publisher
    joy_pub = rospy.Publisher(config['vehicle_control_topic'], Control, queue_size = 10)
    joy_data = Control()

    print('\nStart running. Vroom. Vroom. Vroooooom......')
    print('steer \tthrt: \tbrake \tvelocity')

    while not rospy.is_shutdown():

        if neural_control.image_processed is False:
            continue
        
        # predicted steering angle from an input image
        prediction = neural_control.drive.run(neural_control.image)
        joy_data.steer = prediction

        #############################
        ## TODO: you need to change the vehicle speed wisely  
        ## e.g. not too fast in a curved road and not too slow in a straight road
        
        # if brake is not already applied and sharp turn
        if neural_control.braking is False: 
            if abs(joy_data.steer) > config['sharp_turn_min']: 
                joy_data.throttle = config['throttle_sharp_turn']
                joy_data.brake = 0.5
                neural_control.braking = True
                timer = threading.Timer(config['brake_apply_sec'], neural_control.timer_cb) 
                timer.start()
            else:
                joy_data.throttle = config['throttle_default']
                joy_data.brake = 0
        
            
        ## publish joy_data
        joy_pub.publish(joy_data)

        ## print out
        if config['lstm'] is True:
            cur_output = '{0:.3f} \t{1:.2f} \t{2:.2f} \t{3}\r'.format(prediction[0][0][0], 
                          joy_data.throttle, joy_data.brake, velocity)
        else:
            cur_output = '{0:.3f} \t{1:.2f} \t{2:.2f} \t{3}\r'.format(prediction[0][0], 
                          joy_data.throttle, joy_data.brake, velocity)

        sys.stdout.write(cur_output)
        sys.stdout.flush()
        
        ## ready for processing a new input image
        neural_control.image_processed = False
        neural_control.rate.sleep()



if __name__ == "__main__":
    try:
        if len(sys.argv) != 2:
            exit('Usage:\n$ rosrun run_neural run_neural.py weight_file_name')

        main(sys.argv[1])

    except KeyboardInterrupt:
        print ('\nShutdown requested. Exiting...')
        
