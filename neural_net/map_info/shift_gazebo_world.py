#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun 21 Mar 21∶21∶27 2021
History:

@author: Donghyun Kim
"""

import sys, os, math
from progressbar import ProgressBar

class shift_world():
    def __init__(self):
        self.load_path = sys.argv[1]
            
    def edit_class(self):
        rename = self.load_path.split(".world")
        fr = open(rename[0]+".world", mode='r')
        fw = open(rename[0]+"_shift"+".world", mode='w')
        lines = fr.readlines()
        for i in range(0, len(lines)):
            # word = lines[i].split("<")
            # print(word[0])
            if "pose frame" in lines[i]:
                pose = lines[i].split("<pose frame=''>")
                pose2 = pose[1].split(" ")
                # print(pose[1])
                print("origin :", pose2)
                pose2[0] = str(float(pose2[0]) + float(sys.argv[2]) )
                pose2[1] = str(float(pose2[1]) + float(sys.argv[3]) )
                print("replace:", pose2[0], pose2[1])
                
                pose[1] = pose2[0] + " " + pose2[1]+ " " + pose2[2]+ " " + pose2[3]+ " " + pose2[4]+ " " + pose2[5]
                pose[0] = "      <pose frame=''>"
                fw.write(pose[0]+pose[1])
                # print("hi")
            else:
                fw.write(lines[i])
                
        print("file saving complete")
        fw.close()
        fr.close()
        
        
if __name__ == '__main__':
    
    if len(sys.argv) < 4:
        print('Usage: ')
        exit('$ python shift_gazebo_world.py world_path shift_x shift_y')
    
    m = shift_world()
    m.edit_class()