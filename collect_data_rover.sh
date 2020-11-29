#!/bin/bash

source setup.bash

rosparam set path_to_e2e_data $(pwd)/e2e_rover_data  # /path/to/data

rosrun data_collection data_collection.py $1 
