#!/bin/bash

####
## Assumption: you're at the 'oscar' directory.

##
# oscar main folder location
export OSCAR_PATH=$(pwd)

##
# set up catkin_ws including model path and setup.bash
export GAZEBO_MODEL_PATH=$(pwd)/catkin_ws/src/rover/models:${GAZEBO_MODEL_PATH} > /dev/null 2>&1
source ./catkin_ws/devel/setup.bash

##
# set up sitl_gazebo in PX4 Firmware
source $(pwd)/PX4-Autopilot/Tools/setup_gazebo.bash $(pwd)/PX4-Autopilot $(pwd)/PX4-Autopilot/build/px4_sitl_default > /dev/null 2>&1

export ROS_PACKAGE_PATH=$ROS_PACKAGE_PATH:$(pwd)/PX4-Autopilot > /dev/null 2>&1
export ROS_PACKAGE_PATH=$ROS_PACKAGE_PATH:$(pwd)/PX4-Autopilot/Tools/sitl_gazebo > /dev/null 2>&1

##
# add neural_net folder to PYTHONPATH
export PYTHONPATH=$PYTHONPATH:$(pwd)/neural_net
