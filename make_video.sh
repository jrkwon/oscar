#!/bin/bash

if [ -z "$1" ]; then
    echo "Specify a data path where images are located..."  # modified DataSpeed track
else 
    rm $1.mp4
    cat $1/*.jpg | ffmpeg -f image2pipe -framerate 13 -i - $1.mp4
fi