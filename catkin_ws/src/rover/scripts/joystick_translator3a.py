#!/usr/bin/env python

import rospy
from rover.msg import Control
from sensor_msgs.msg import Joy
from geometry_msgs.msg import Twist

from config import Config



# joystick -0.924-0.935


# Steering
STEERING_AXIS = 0   # left 1 --> center 0 --> right -1 roll

# Speed control
BUTTON_A = 1        # speed step down
BUTTON_B = 2        # steering step up
BUTTON_X = 0        # steering step down
BUTTON_Y = 3        # speed step up


# Throttle and Brake
THROTTLE_AXIS = 3   # release -1 --> press 1
#BRAKE_AXIS = 1      # release -1 --> press 1
#BRAKE_POINT = -0.9  # consider brake is applied if value is greater than this.

# Gear shift
# to be neutral, bothe SHIFT_FORWARD & SHIFT_REVERSE must be 0
SHIFT_FORWARD = 6     # forward 1
SHIFT_REVERSE = 7     # reverse 1

# Max speed and steering factor
MAX_THROTTLE_FACTOR = 1
MAX_STEERING_FACTOR = 0.2
# Default speed and steering factor : 
INIT_THROTTLE_FACTOR = 1
INIT_STERRING_FACTOR = 0.2

# Small value
SMALL_VALUE = 0.0001

class Translator:
    def __init__(self):
        self.sub = rospy.Subscriber("joy", Joy, self.callback)
        self.pub = rospy.Publisher('rover', Control, queue_size=20)
        self.pub4mavros = rospy.Publisher('/mavros/setpoint_velocity/cmd_vel_unstamped', Twist, queue_size=80)

        self.last_published_time = rospy.get_rostime()
        self.last_published = None
        self.timer = rospy.Timer(rospy.Duration(1./20.), self.timer_callback)

        self.throttle_factor = INIT_THROTTLE_FACTOR
        self.steering_factor = INIT_STERRING_FACTOR
        
    def timer_callback(self, event):
        if self.last_published and self.last_published_time < rospy.get_rostime() + rospy.Duration(1.0/20.):
            self.callback(self.last_published)

    def callback(self, message):
        rospy.logdebug("joy_translater received axes %s",message.axes)
        command = Control()
        command.header = message.header

        # Throttle speed control
        if message.buttons[BUTTON_Y] == 1:
            self.throttle_factor = self.throttle_factor + 1 if self.throttle_factor < MAX_THROTTLE_FACTOR else MAX_THROTTLE_FACTOR
        if message.buttons[BUTTON_A] == 1:
            self.throttle_factor = self.throttle_factor - 1 if self.throttle_factor > 1 else 1

        # Steering speed control
        if message.buttons[BUTTON_B] == 1:
            self.steering_factor = self.steering_factor + 1 if self.steering_factor < MAX_STEERING_FACTOR else MAX_STEERING_FACTOR
        if message.buttons[BUTTON_X] == 1:
            self.steering_factor = self.sterring_factor - 1 if self.steering_factor > 1 else 1

        #if message.axes[BRAKE_AXIS] > BRAKE_POINT:
		    #command.brake = 1.0
        
        # Note: init value of axes are all zeros
        # --> problem with -1 to 1 range values like brake
        #if message.axes[BRAKE_AXIS] > -1*SMALL_VALUE and message.axes[BRAKE_AXIS] < SMALL_VALUE:
		    #command.brake = 0.0

        if message.axes[THROTTLE_AXIS] >= 0:
            command.throttle = message.axes[THROTTLE_AXIS]
            #command.brake = 0.0
        else:
            command.throttle = 0.0
        
        if message.buttons[SHIFT_FORWARD] == 1:
            command.shift_gears = Control.FORWARD
        elif message.buttons[SHIFT_REVERSE] == 1:
            command.shift_gears = Control.REVERSE
        elif message.buttons[SHIFT_FORWARD] == 0 and message.buttons[SHIFT_REVERSE] == 0 :
            command.shift_gears = Control.NEUTRAL
        else:
            command.shift_gears = Control.NO_COMMAND

        command.steer = message.axes[STEERING_AXIS]
        command.throttle = message.axes[THROTTLE_AXIS]

        # scale throttle and steering 
        command.throttle = command.throttle*self.throttle_factor
        command.steer = command.steer*self.steering_factor

        command4mavros = Twist()
        if command.shift_gears == Control.FORWARD:
            command4mavros.linear.y = command.throttle
            command4mavros.linear.x = command.steer
        elif command.shift_gears == Control.REVERSE:
            command4mavros.linear.y = -command.throttle
            command4mavros.linear.x = command.steer
        else:
            command4mavros.linear.x = command.steer
            command4mavros.linear.y = command.throttle

        self.pub4mavros.publish(command4mavros)
        
        self.last_published = message
        self.pub.publish(command)

if __name__ == '__main__':
    rospy.init_node('joystick_translator3')
    t = Translator()
    rospy.spin()
