#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Sun 21 Mar 21∶21∶27 2021
History:

@author: Donghyun Kim
"""

import rospy
from gazebo_msgs.msg import ModelStates

import sys, os, math
from progressbar import ProgressBar

class MapInfoGenerator:
    def __init__(self):
        self.sub = rospy.Subscriber("/gazebo/model_states", ModelStates, self._callback)
        self.msg_flag = False
        self.msg_track_name = []
        # self.msg_track_pose = []
        self.track = []
        self.txt = []
        self.oscar_path = str()
        self.count = 0
        
        ########################
        #cal groundtruth
        self.quadrant = 1
        self.last_position = [0,0]
        self.total_error = 0
        
    def _callback(self, msg):
        self.msg_track_name = msg.name
        if len(self.msg_track_name) > 0 and self.msg_flag == False and self.count==0:
            self._track_check()
            
            
    
    def _oscar_path_check(self):
        current_path = os.getcwd().split("/")
        check = False
        for i in range(0, len(current_path)):
            if current_path[i] == "oscar":
                self.oscar_path += current_path[i] +"/"
                check = True
            else:
                self.oscar_path += current_path[i] +"/"
    
    def _track_check(self):
        self.count += 1
        track_form=[]
        track_length=[]
        for i in range(0, len(self.msg_track_name)):
            if 'Road' in self.msg_track_name[i]:
                track_form.append(self.msg_track_name[i].split(' ')[1])
                track_length.append( (self.msg_track_name[i].split(' ')[4]).split('m')[0] )
        self.track.append(track_form)
        self.track.append(track_length)
        self._oscar_path_check()
        self._build_txt(self.oscar_path+"/neural_net/map_info", self.track)
        self._cal_groundtruth(self.oscar_path+"/neural_net/map_info")
        os.system("rosnode kill " + "mapinfo_generator")
        
    def _build_txt(self, data_path, data):
        if data_path[-1] != '/':
            data_path = data_path + '/'

        # find the second '/' from the end to get the folder name
        loc_dir_delim = data_path[:-1].rfind('/')
        if (loc_dir_delim != -1):
            folder_name = data_path[loc_dir_delim+1:-1]
            txt_file = folder_name + ".txt"
        else:
            folder_name = data_path[:-1]
            txt_file = folder_name + ".txt"
        new_txt = []
        bar = ProgressBar()
        for i in bar(range(len(data))):
            new_txt.append(str(data[i])
                        + '\n')
            
        new_txt_fh = open(data_path + txt_file, 'w')
        for i in range(len(new_txt)):
            new_txt_fh.write(new_txt[i])
        new_txt_fh.close()
        
    def _cal_groundtruth(self, data_path):
        new_csv = []
        for i in range(0, len(self.track[0])):
            new_csv.append(self.track[0][i]+" "+self.track[1][i]+"m"+'\n')
            if self.track[0][i] == "Straight":
                road_dist = float(self.track[1][i]) #도로의 길이 (m). 
                data_term = 1                       #1cm 마다 pose를 구할것임
                for _ in range(0, int(road_dist/data_term)):
                    if self.quadrant == 1:
                        self.last_position = [self.last_position[0]+data_term , self.last_position[1]          ]
                    elif self.quadrant == 2:
                        self.last_position = [self.last_position[0]           , self.last_position[1]+data_term]
                    elif self.quadrant == 3:
                        self.last_position = [self.last_position[0]-data_term , self.last_position[1]          ]
                    elif self.quadrant == 4:
                        self.last_position = [self.last_position[0]           , self.last_position[1]-data_term]
                    # print(self.last_position)
                    for n in range(len(self.last_position)):
                        if self.last_position[n] < 0.0001 and self.last_position[n] > -0.0001:
                            self.last_position[n] = 0
                    new_csv.append(str(self.last_position[0])+" "+str(self.last_position[1])+'\n')
                
            elif self.track[0][i] == "Left" or self.track[0][i] == "Right":
                road_dist = float(self.track[1][i]) #도로의 길이 (m). 
                data_term = 1                       #1cm 마다 pose를 구할것임
                theta = math.pi / (2 * (road_dist/data_term))
                print(theta)
                rotate_result = [0,0]
                rotate_position = [road_dist, 0]
                pose = [0, 0]
                if self.track[0][i] == "Left":
                    for j in range(0, int(road_dist/data_term)):
                        rotate_result[0] = rotate_position[0]*math.cos((j+1)*theta) - rotate_position[1]*math.sin((j+1)*theta)
                        rotate_result[1] = rotate_position[0]*math.sin((j+1)*theta) + rotate_position[1]*math.cos((j+1)*theta)
                        
                        print("rotate result : ",rotate_result)
                        print("last position : ",self.last_position)
                        if self.quadrant == 1:
                            pose = [self.last_position[0] + rotate_result[1]               , self.last_position[1] + (road_dist - rotate_result[0])]
                        elif self.quadrant == 2:
                            pose = [self.last_position[0] - (road_dist - rotate_result[0]) , self.last_position[1] + rotate_result[1]              ]
                        elif self.quadrant == 3:
                            pose = [self.last_position[0] - rotate_result[1]               , self.last_position[1] - (road_dist - rotate_result[0])]
                        elif self.quadrant == 4:
                            pose = [self.last_position[0] + (road_dist - rotate_result[0]) , self.last_position[1] - rotate_result[1]              ]
                        
                        for n in range(len(pose)):
                            if pose[n] < 0.0001 and pose[n] > -0.0001:
                                pose[n] = 0
                        new_csv.append(str(pose[0])+" "+str(pose[1])+'\n')
                        
                    if self.quadrant == 1:
                        self.last_position = [self.last_position[0] + rotate_result[1]               , self.last_position[1] + (road_dist - rotate_result[0])]
                    elif self.quadrant == 2:
                        self.last_position = [self.last_position[0] - (road_dist - rotate_result[0]) , self.last_position[1] + rotate_result[1]              ]
                    elif self.quadrant == 3:
                        self.last_position = [self.last_position[0] - rotate_result[1]               , self.last_position[1] - (road_dist - rotate_result[0])]
                    elif self.quadrant == 4:
                        self.last_position = [self.last_position[0] + (road_dist - rotate_result[0]) , self.last_position[1] - rotate_result[1]              ]
                        
                    self.quadrant += 1
                    
                elif self.track[0][i] == "Right":
                    for j in range(0, int(road_dist/data_term)):
                        rotate_result[0] = rotate_position[0]*math.cos((j+1)*theta) - rotate_position[1]*math.sin((j+1)*theta)
                        rotate_result[1] = rotate_position[0]*math.sin((j+1)*theta) + rotate_position[1]*math.cos((j+1)*theta)
                        
                        print("rotate result : ",rotate_result)
                        print("last position : ",self.last_position)
                        if self.quadrant == 1:
                            pose = [self.last_position[0] + rotate_result[1]               , self.last_position[1] - (road_dist - rotate_result[0])]
                        elif self.quadrant == 2:
                            pose = [self.last_position[0] + (road_dist - rotate_result[0]) , self.last_position[1] + rotate_result[1]              ]
                        elif self.quadrant == 3:
                            pose = [self.last_position[0] - rotate_result[1]               , self.last_position[1] + (road_dist - rotate_result[0])]
                        elif self.quadrant == 4:
                            pose = [self.last_position[0] - (road_dist - rotate_result[0]) , self.last_position[1] - rotate_result[1]              ]

                        for n in range(len(pose)):
                            if pose[n] < 0.0001 and pose[n] > -0.0001:
                                pose[n] = 0
                        new_csv.append(str(pose[0])+" "+str(pose[1])+'\n')
                        
                    if self.quadrant == 1:
                        self.last_position = [self.last_position[0] + rotate_result[1]               , self.last_position[1] - (road_dist - rotate_result[0])]
                    elif self.quadrant == 2:
                        self.last_position = [self.last_position[0] + (road_dist - rotate_result[0]) , self.last_position[1] + rotate_result[1]              ]
                    elif self.quadrant == 3:
                        self.last_position = [self.last_position[0] - rotate_result[1]               , self.last_position[1] + (road_dist - rotate_result[0])]
                    elif self.quadrant == 4:
                        self.last_position = [self.last_position[0] - (road_dist - rotate_result[0]) , self.last_position[1] - rotate_result[1]              ]
                            
                        
                    self.quadrant -= 1
                if self.quadrant == 0:
                    self.quadrant = 4
                elif self.quadrant == 5:
                    self.quadrant = 1
        
        # write a new csv
        new_csv_fh = open(data_path + "/ground_truth.csv", 'w')
        for i in range(len(new_csv)):
            new_csv_fh.write(new_csv[i])
        new_csv_fh.close()
        
    def _cal_roadformula(self, car_pose):
        last_position = [0, 0]
        for i in range(0, len(self.track[0])):
            road_dist = self.track[1][i]
            road_box = [0, 0, 0]
            margin = 50.0
            error = 0
            if self.track[0][i] == "Straight":
                start_pnt = last_position
                if self.quadrant == 1:
                    end_pnt = [start_pnt[0]+road_dist, start_pnt[1]]
                    road_box = [ [start_pnt[0]  , start_pnt[1]-margin] 
                               , [end_pnt[0]    , end_pnt[1]-margin] 
                               , [start_pnt[0]  , start_pnt[1]+margin] ]
                    if road_box[0][0] < car_pose[0] < road_box[1][0] and road_box[0][1] < car_pose[1] < road_box[2][1]:
                        error = abs(start_pnt[1] - car_pose[1])
                elif self.quadrant == 2:
                    end_pnt = [start_pnt[0], start_pnt[1]+road_dist]
                    road_box = [ [start_pnt[0]-margin  , start_pnt[1]] 
                               , [start_pnt[0]+margin  , start_pnt[1]] 
                               , [end_pnt[0]-margin    , end_pnt[1]] ]
                    if road_box[0][0] < car_pose[0] < road_box[1][0] and road_box[0][1] < car_pose[1] < road_box[2][1]:
                        error = abs(start_pnt[0] - car_pose[0])
                elif self.quadrant == 3:
                    end_pnt = [start_pnt[0]-road_dist, start_pnt[1]]
                    road_box = [ [end_pnt[0]    , end_pnt[1]-margin] 
                               , [start_pnt[0]  , start_pnt[1]-margin] 
                               , [end_pnt[0]    , end_pnt[1]+margin] ]
                    if road_box[0][0] < car_pose[0] < road_box[1][0] and road_box[0][1] < car_pose[1] < road_box[2][1]:
                        error = abs(start_pnt[1] - car_pose[1])
                elif self.quadrant == 4:
                    end_pnt = [start_pnt[0], start_pnt[1]-road_dist]
                    road_box = [ [end_pnt[0]-margin     , end_pnt[1]] 
                               , [end_pnt[0]+margin     , end_pnt[1]] 
                               , [start_pnt[0]-margin   , start_pnt[1]] ]
                    if road_box[0][0] < car_pose[0] < road_box[1][0] and road_box[0][1] < car_pose[1] < road_box[2][1]:
                        error = abs(start_pnt[0] - car_pose[0])
                print(error)
                last_position = end_pnt
            
            elif self.track[0][i] == "Left":
                start_pnt = last_position
                if self.quadrant == 1:
                    end_pnt = [start_pnt[0]+road_dist, start_pnt[1]+road_dist]
                    center = [start_pnt[0], start_pnt[1]+road_dist]
                    if start_pnt[0] < car_pose[0] < end_pnt[0]+margin and start_pnt[1]-margin < car_pose[1] < end_pnt[1]:
                        dx = car_pose[0]-center[0]
                        dy = car_pose[1]-center[1]
                        error = abs(math.sqrt( (dx**2) + (dy**2) ) - road_dist)
                    
                elif self.quadrant == 2:
                    end_pnt = [start_pnt[0]-road_dist, start_pnt[1]+road_dist]
                    center = [start_pnt[0]-road_dist, start_pnt[1]]
                    if end_pnt[0] < car_pose[0] < start_pnt[0]+margin and start_pnt[1] < car_pose[1] < end_pnt[1]+margin:
                        dx = car_pose[0]-center[0]
                        dy = car_pose[1]-center[1]
                        error = abs(math.sqrt( (dx**2) + (dy**2) ) - road_dist)
                        
                elif self.quadrant == 3:
                    end_pnt = [start_pnt[0]-road_dist, start_pnt[1]-road_dist]
                    center = [start_pnt[0], start_pnt[1]-road_dist]
                    if end_pnt[0]-margin < car_pose[0] < start_pnt[0] and end_pnt[1] < car_pose[1] < start_pnt[1]+margin:
                        dx = car_pose[0]-center[0]
                        dy = car_pose[1]-center[1]
                        error = abs(math.sqrt( (dx**2) + (dy**2) ) - road_dist)
                        
                elif self.quadrant == 4:
                    end_pnt = [start_pnt[0]+road_dist, start_pnt[1]-road_dist]
                    center = [start_pnt[0]+road_dist, start_pnt[1]]
                    if start_pnt[0]-margin < car_pose[0] < end_pnt[0] and end_pnt[1]-margin < car_pose[1] < start_pnt[1]:
                        dx = car_pose[0]-center[0]
                        dy = car_pose[1]-center[1]
                        error = abs(math.sqrt( (dx**2) + (dy**2) ) - road_dist)
                last_position = end_pnt
                
            elif self.track[0][i] == "Right":
                start_pnt = last_position
                if self.quadrant == 1:
                    end_pnt = [start_pnt[0]+road_dist, start_pnt[1]-road_dist]
                    center = [start_pnt[0], start_pnt[1]-road_dist]
                    if start_pnt[0] < car_pose[0] < end_pnt[0]+margin and end_pnt[1] < car_pose[1] < start_pnt[1]+margin:
                        dx = car_pose[0]-center[0]
                        dy = car_pose[1]-center[1]
                        error = abs(math.sqrt( (dx**2) + (dy**2) ) - road_dist)
                elif self.quadrant == 2:
                    end_pnt = [start_pnt[0]+road_dist, start_pnt[1]+road_dist]
                    center = [start_pnt[0]+road_dist, start_pnt[1]]
                    if start_pnt[0]-margin < car_pose[0] < end_pnt[0] and start_pnt[1] < car_pose[1] < end_pnt[1]+margin:
                        dx = car_pose[0]-center[0]
                        dy = car_pose[1]-center[1]
                        error = abs(math.sqrt( (dx**2) + (dy**2) ) - road_dist)
                elif self.quadrant == 3:
                    end_pnt = [start_pnt[0]-road_dist, start_pnt[1]+road_dist]
                    center = [start_pnt[0], start_pnt[1]+road_dist]
                    if end_pnt[0]-margin < car_pose[0] < start_pnt[0] and start_pnt[1]-margin < car_pose[1] < end_pnt[1]:
                        dx = car_pose[0]-center[0]
                        dy = car_pose[1]-center[1]
                        error = abs(math.sqrt( (dx**2) + (dy**2) ) - road_dist)
                elif self.quadrant == 4:
                    end_pnt = [start_pnt[0]-road_dist, start_pnt[1]-road_dist]
                    center = [start_pnt[0]-road_dist, start_pnt[1]]
                    if end_pnt[0] < car_pose[0] < start_pnt[0]+margin and end_pnt[1]-margin < car_pose[1] < start_pnt[1]:
                        dx = car_pose[0]-center[0]
                        dy = car_pose[1]-center[1]
                        error = abs(math.sqrt( (dx**2) + (dy**2) ) - road_dist)
                last_position = end_pnt
            
            
            self.total_error += error
                        
                
        
        
def main():
    rospy.init_node('mapinfo_generator')
    m = MapInfoGenerator()
    rospy.spin()
        
        
if __name__ == '__main__':
    main()