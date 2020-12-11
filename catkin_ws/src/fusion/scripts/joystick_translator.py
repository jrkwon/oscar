#!/usr/bin/env python

# Copyright 2017 Open Source Robotics Foundation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import rospy
from fusion.msg import Control
from sensor_msgs.msg import Joy

#######################################
## Logitech G920 with Pedal and Shift

# Steering
STEERING_AXIS = 0   # left 1 --> center 0 --> right -1

# Speed control
BUTTON_A = 0        # speed step down
BUTTON_B = 1        # steering step up
BUTTON_X = 2        # steering step down
BUTTON_Y = 3        # speed step up


# Throttle and Brake
THROTTLE_AXIS = 1   # release -1 --> press 1
BRAKE_AXIS = 2      # release -1 --> press 1
BRAKE_POINT = -0.9  # consider brake is applied if value is greater than this.

# Gear shift
# to be neutral, bothe SHIFT_FORWARD & SHIFT_REVERSE must be 0
SHIFT_FORWARD = 14     # forward 1
SHIFT_REVERSE = 15     # reverse 1

# Max speed and steering factor
MAX_THROTTLE_FACTOR = 10
MAX_STEERING_FACTOR = 5
# Default speed and steering factor
INIT_THROTTLE_FACTOR = 3
INIT_STERRING_FACTOR = 1

# Small value
SMALL_VALUE = 0.0001

class Translator:
    def __init__(self):
        self.sub = rospy.Subscriber("joy", Joy, self.callback)
        self.pub = rospy.Publisher('fusion', Control, queue_size=1)
        self.last_published_time = rospy.get_rostime()
        self.last_published = None
        self.timer = rospy.Timer(rospy.Duration(1./20.), self.timer_callback)
        
    def timer_callback(self, event):
        if self.last_published and self.last_published_time < rospy.get_rostime() + rospy.Duration(1.0/20.):
            self.callback(self.last_published)

    def callback(self, message):
        rospy.logdebug("joy_translater received axes %s",message.axes)
        command = Control()
        command.header = message.header

        if message.axes[BRAKE_AXIS] > BRAKE_POINT:
		    command.brake = 1.0

        # Note: init value of axes are all zeros
        # --> problem with -1 to 1 range values like brake
        if message.axes[BRAKE_AXIS] > -1*SMALL_VALUE and message.axes[BRAKE_AXIS] < SMALL_VALUE:
		    command.brake = 0.0

        if message.axes[THROTTLE_AXIS] >= 0:
            command.throttle = message.axes[THROTTLE_AXIS]
            command.brake = 0.0
        else:
            command.throttle = 0.0

        if message.buttons[SHIFT_FORWARD] == 1:
            command.shift_gears = Control.FORWARD
        elif message.buttons[SHIFT_FORWARD] == 0 and message.buttons[SHIFT_REVERSE] == 0 :
            command.shift_gears = Control.NEUTRAL
        elif message.buttons[SHIFT_REVERSE] == 1:
            command.shift_gears = Control.REVERSE
        else:
            command.shift_gears = Control.NO_COMMAND

        command.steer = message.axes[STEERING_AXIS]
        self.last_published = message
        self.pub.publish(command)

if __name__ == '__main__':
    rospy.init_node('joystick_translator')
    t = Translator()
    rospy.spin()
