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
from mavros_msgs.msg import AttitudeTarget
from nav_msgs.msg import Odometry
import math
from rover.msg import Control
import sys
import os

import const
from image_converter import ImageConverter
from drive_run import DriveRun
from config import Config
from image_process import ImageProcess

config = Config.neural_net
velocity = 0
def pos_vel_cb(value):
	global velocity

	vel_x = value.twist.twist.linear.x 
	vel_y = value.twist.twist.linear.y
	vel_z = value.twist.twist.linear.z
		
	velocity = math.sqrt(vel_x**2 + vel_y**2 + vel_z**2)
class NeuralControl:
    def __init__(self):
        
        self.ic = ImageConverter()
        self.image_process = ImageProcess()
        self.rate = rospy.Rate(10000000)
        self.drive= DriveRun(weight_file_name)
        rospy.Subscriber(Config.data_collection['camera_image_topic'], Image, self._controller_cb)
        self.image = None
        self.image_processed = False
        #self.config = Config()
        self.braking = False
        
        self.joy_pub = rospy.Publisher(Config.data_collection['vehicle_control_topic'], Control, queue_size = 20)
        self.joy_pub4mavros2 = rospy.Publisher('/mavros/setpoint_raw/attitude', AttitudeTarget, queue_size=100)
        self.last_published_time = rospy.get_rostime()
        self.last_published = None
        self.timer = rospy.Timer(rospy.Duration(1./20.), self.timer_callback)

    def timer_callback(self, event):
        if self.last_published and self.last_published_time < rospy.get_rostime() + rospy.Duration(1.0/20.):
            self.callback(self.last_published)
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
        

      


	  
    def callback(self, weight_file_name):
    	gpu_options.set()

	# ready for neural network
	#neural_control = nc(self,weight_file_name)
			
	
	# ready for /bolt topic publisher
	joy_data = Control()

	
	joy_data4mavros2 = AttitudeTarget()
	self.joy_data.throttle= Config.run_neural['throttle_default']
	self.joy_pub4mavros2.publish(joy_data4mavros2)


	print('\nStart running. Vroom. Vroom. Vroooooom......')
	print('steer \tthrt: \tbrake \tvelocity')

	self.prediction = self.neural_control.drive.run((neural_control.image, ))
	joy_data.steer = prediction[0][0]

		##############################    
		## publish mavros control topic
		

	joy_data4mavros2.thrust = Config.run_neural['throttle_default']
	if joy_data.steer >= 0:
		joy_data4mavros2.orientation.x = joy_data.steer
	else:
		joy_data4mavros2.orientation.y = joy_data.steer


	self.joy_pub.publish(joy_data)
	self.joy_pub4mavros2.publish(joy_data4mavros2)

        self.last_published = joy_data
        self.joy_pub4mavros2.publish(joy_data4mavros2)

	## print out
	self.cur_output = '{0:.3f} \t{1:.3f} \t{2:.3f} \t{3:.3f}\r'.format(joy_data.steer, 
		                  joy_data.throttle, joy_data.brake, velocity)

	sys.stdout.write(self.cur_output)
	sys.stdout.flush()

	## ready for processing a new input image
	self.neural_control.image_processed = False
	#neural_control.rate.sleep()


if __name__ == "__main__":
    	
    rospy.init_node('run_neural')
    rospy.Subscriber(Config.data_collection['base_pose_topic'], Odometry, pos_vel_cb)
    joy_pub4mavros2 = rospy.Publisher('/mavros/setpoint_raw/attitude', AttitudeTarget, queue_size=100)
    joy_data4mavros2 = AttitudeTarget()
    joy_data4mavros2.thrust=0.5
    joy_pub4mavros2.publish(joy_data4mavros2)
    r = rospy.Rate(30)
    weight_file_name = 'e2e_data/9/trainfolder/2022-08-15-16-54-34_rover_template_N0'
    nc = NeuralControl()
    while not rospy.is_shutdown():
    	joy_pub4mavros2.publish(joy_data4mavros2)
    	r.sleep()

        
