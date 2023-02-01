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
from std_msgs.msg import Int32
from sensor_msgs.msg import Image
from nav_msgs.msg import Odometry
import math

import sys
import os

import argparse

import const
from image_converter import ImageConverter
from drive_run import DriveRun
from config import Config
from image_process import ImageProcess

if Config.data_collection['vehicle_name'] == 'fusion':
    from fusion.msg import Control
elif Config.data_collection['vehicle_name'] == 'rover':
    from geometry_msgs.msg import Twist
    from rover.msg import Control
else:
    exit(Config.data_collection['vehicle_name'] + 'not supported vehicle.')

import gpu_options

config = Config.neural_net
velocity = 0

class NeuralControl:
    def __init__(self, weight_file_name):
        rospy.init_node('run_neural')
        self.ic = ImageConverter()
        self.image_process = ImageProcess()
        self.rate = rospy.Rate(30)
        self.drive= DriveRun(weight_file_name)
        rospy.Subscriber(Config.data_collection['camera_image_topic'], Image, self._controller_cb)
        self.image = None
        self.image_processed = False
        #self.config = Config()
        self.braking = False

    def _controller_cb(self, image): 
        img = self.ic.imgmsg_to_opencv(image)
        cropped = img[Config.data_collection['image_crop_y1']:Config.data_collection['image_crop_y2'],
                      Config.data_collection['image_crop_x1']:Config.data_collection['image_crop_x2']]
                      
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
        
    def _timer_cb(self):
        self.braking = False

    def apply_brake(self):
        self.braking = True
        timer = threading.Timer(Config.run_neural['brake_apply_sec'], self._timer_cb) 
        timer.start()

      
def pos_vel_cb(value):
    global velocity

    vel_x = value.twist.twist.linear.x 
    vel_y = value.twist.twist.linear.y
    vel_z = value.twist.twist.linear.z
    
    velocity = math.sqrt(vel_x**2 + vel_y**2 + vel_z**2)


def calc_weights(speed):
    w1 = -0.4 * np.tanh(2.5 * speed - 50) + 0.5
    w2 = 1 - w1
    return w1, w2

def main(args):

    gpu_options.set()

    base_model = args.base_model
    second_model = args.second_model
    # print(base_model)
    # print(second_model)

    # ready for neural network
    neural_control_base = NeuralControl(base_model)
    neural_control_second = NeuralControl(second_model)

    # print('-------------------------------')
    # print(neural_control_base==neural_control_second) # --> false
    # # print(neural_control_second)
    # print('-------------------------------')

    rospy.Subscriber(Config.data_collection['base_pose_topic'], Odometry, pos_vel_cb)
    # ready for /bolt topic publisher
    joy_pub = rospy.Publisher(Config.data_collection['vehicle_control_topic'], Control, queue_size = 10)
    joy_data = Control()
    # print(joy_data)
    joy_data_base = Control()
    joy_data_second = Control()

    if Config.data_collection['vehicle_name'] == 'rover':
        joy_pub4mavros = rospy.Publisher(Config.config['mavros_cmd_vel_topic'], Twist, queue_size=20)

    print('\nStart running. Vroom. Vroom. Vroooooom......')
    print('steer \tthrt \tbrake \tvelocity \tBM_weight \tDCM_weight')

    use_predicted_throttle = True if config['num_outputs'] == 2 else False
    while not rospy.is_shutdown():
        
        # if neural_control.image_processed is False:
        #     continue

        if neural_control_base.image_processed is False:
            continue

        if neural_control_second.image_processed is False:
            continue
        
        # predicted steering angle from an input image
        if config['num_inputs'] == 2:
            # prediction = neural_control.drive.run((neural_control.image, velocity))
            prediction_base = neural_control_base.drive.run((neural_control_base.image, velocity))
            prediction_second = neural_control_second.drive.run((neural_control_second.image, velocity))
            if config['num_outputs'] == 2:
                # prediction is [ [] ] numpy.ndarray
                joy_data.steer = prediction_base[0][0]
                joy_data.throttle = prediction_base[0][1]

                joy_data_base.steer = prediction_base[0][0]
                joy_data_base.throttle = prediction_base[0][1]

                joy_data_second.steer = prediction_second[0][0]
                joy_data_second.throttle = prediction_second[0][1]
            else: # num_outputs is 1
                joy_data.steer = prediction_base[0][0]
                joy_data_base.steer = prediction_base[0][0]
                joy_data_second.steer = prediction_second[0][0]

        else: # num_inputs is 1
            # prediction = neural_control.drive.run((neural_control.image, ))
            prediction_base = neural_control_base.drive.run((neural_control_base.image, ))
            # print(prediction_base)
            prediction_second = neural_control_second.drive.run((neural_control_second.image, ))
            if config['num_outputs'] == 2:
                # prediction is [ [] ] numpy.ndarray
                joy_data.steer = prediction_base[0][0]
                joy_data.throttle = prediction_base[0][1]

                joy_data_base.steer = prediction_base[0][0]
                joy_data_base.throttle = prediction_base[0][1]

                joy_data_second.steer = prediction_second[0][0]
                joy_data_second.throttle = prediction_second[0][1]
            else: # num_outputs is 1
                joy_data.steer = prediction_base[0][0]
                joy_data_base.steer = prediction_base[0][0]
                joy_data_second.steer = prediction_second[0][0]
            
        #############################
        ## very very simple controller
        ## 

        is_sharp_turn = False
        # if brake is not already applied and sharp turn
        if neural_control_base.braking is False: 
            if velocity < Config.run_neural['velocity_0']: # too slow then no braking
                joy_data.throttle = Config.run_neural['throttle_default'] # apply default throttle
                joy_data.brake = 0
            elif abs(joy_data.steer) > Config.run_neural['sharp_turn_min']:
                is_sharp_turn = True
            
            if is_sharp_turn or velocity > Config.run_neural['max_vel']: 
                joy_data.throttle = Config.run_neural['throttle_sharp_turn']
                joy_data.brake = Config.run_neural['brake_val']
                neural_control_base.apply_brake()
            else:
                if use_predicted_throttle is False:
                    joy_data.throttle = Config.run_neural['throttle_default']
                joy_data.brake = 0

        w_base, w_second = calc_weights(velocity)
        # print ('w_base = ', w_base)
        # print ('w_second = ', w_second)
        joy_data.steer = w_base*joy_data_base.steer + w_second*joy_data_second.steer

        joy_pub.publish(joy_data)

        ##############################    
        ## publish mavros control topic
        
        if Config.data_collection['vehicle_name'] == 'rover':
            joy_data4mavros = Twist()
            # if neural_control.braking is True:
            if neural_control_base.braking is True:
                joy_data4mavros.linear.x = 0
                joy_data4mavros.linear.y = 0
            else: 
                joy_data4mavros.linear.x = joy_data.throttle*Config.run_neural['scale_factor_throttle']
                joy_data4mavros.linear.y = joy_data.steer*Config.run_neural['scale_factor_steering']

            joy_pub4mavros.publish(joy_data4mavros)


        ## print out
        cur_output = '{0:.3f}\t{1:.3f}\t{2:.3f}\t{3:.3f}         \t{4:.3f}        \t{5:.3f}\r'.format(
                        joy_data.steer, joy_data.throttle, joy_data.brake, velocity, w_base, w_second)

        sys.stdout.write(cur_output)
        sys.stdout.flush()
        
        ## ready for processing a new input image
        # neural_control.image_processed = False
        # neural_control.rate.sleep()
        neural_control_base.image_processed = False
        neural_control_base.rate.sleep()



if __name__ == "__main__":
    try:
        # if len(sys.argv) != 2:
        #     exit('Usage:\n$ rosrun run_neural run_neural.py weight_file_name')
        argparser = argparse.ArgumentParser(
        description='Fusing the outputs of two NN models')
    
        argparser.add_argument(
            '-bm','--base_model',
            help='base model for low velocity',
            default='path/to/base_model',
            type=str)
        
        argparser.add_argument(
            '-pm', '--predictive_model',
            help='second model for higher velocity',
            default='path/to/predictive_model',
            type=str)

        args = argparser.parse_args()

        # main(sys.argv[1])
        main(args)

    except KeyboardInterrupt:
        print ('\nShutdown requested. Exiting...')
        
