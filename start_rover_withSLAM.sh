#!/bin/bash

#cd PX4-Autopilot
#DONT_RUN=1 make px4_sitl gazebo_rover
#cd ..
source ./setup.bash
cd catkin_ws
catkin_make
cd ..

roslaunch rover bridge_mavros2a.launch
