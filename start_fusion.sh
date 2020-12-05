#!/bin/bash

cd catkin_ws
catkin_make
cd ..

source ./catkin_ws/devel/setup.bash

if [ -z "$1" ]; then
    echo "Starting with default world..."  # modified DataSpeed track
    roslaunch fusion sitl.launch x:=7.58439 y:=-102.82 z:=0.005 R:=0 P:=0 Y:=0
elif [ "$1" == "sonoma_raceway" ]; then
    echo "Starting with $1..." #### experimental
    roslaunch fusion sitl.launch world:=$1 x:=0 y:=0 z:=0.179517 R:=-0.001258 P:=-0.010927 Y:=2.4979
elif [ "$1" == "mcity_jaerock" ] ; then
    echo "Starting with $1..." #### experimental
    roslaunch fusion sitl.launch world:=$1 x:=3 y:=-12 z:=0.017607 R:=0 P:=0 Y:=0
elif [ "$1" == "simple_city" ] ; then
    echo "Starting with $1..." #### experimental
    roslaunch fusion sitl.launch world:=$1 
else 
    echo "Error: no $1.world file exist." 
fi