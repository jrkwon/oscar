#!/bin/bash

source setup.bash

rosparam set path_to_e2e_data $(pwd)  # /path/to/data
#rosrun data_collection data_collection.py $1 /camera/image_raw:=/fusion/front_camera/image_raw
rosrun data_collection data_collection.py $1 