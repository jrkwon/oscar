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
import numpy as np
from progressbar import ProgressBar
from drive_data import DriveData

class MapInfoGenerator:
    def __init__(self, csv_path, follow_lane):
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
        ########################
        #cal roadformular
        self.total_error = []
        self.total_error_nabs = []
        self.car_pose = None
        self.car_steering = None
        self.car_velocities = None
        self.car_times = None
        
        self.csv_path = csv_path
        self.follow_lane = follow_lane
        
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
        self._cal_totalerror()
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
        
    def _build_csv(self, data_path, data):
        if data_path[-1] != '/':
            data_path = data_path + '/'

        # find the second '/' from the end to get the folder name
        loc_dir_delim = data_path[:-1].rfind('/')
        if (loc_dir_delim != -1):
            folder_name = data_path[loc_dir_delim+1:-1]
            txt_file = folder_name + ".csv"
        else:
            folder_name = data_path[:-1]
            txt_file = folder_name + ".csv"
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
        
        last_position = [0, 0]
        for i in range(0, len(self.track[0])):
            road_dist = float(self.track[1][i])
            curve_margin = 2.333
            road_box = [0, 0, 0]
            box_margin = 5.0
            # box_margin = 50.0
            sample = 319
            if self.track[0][i] == "Straight":
                for j in range(sample):
                    start_pnt = last_position
                    if self.quadrant == 1:
                        end_pnt = [start_pnt[0]+road_dist, start_pnt[1]]
                        road_box = [ [start_pnt[0]  , start_pnt[1]-box_margin] 
                                , [end_pnt[0]    , end_pnt[1]-box_margin] 
                                , [start_pnt[0]  , start_pnt[1]+box_margin] ]
                        new_csv.append(str(start_pnt[0]+road_dist*(j+1)/sample)+', '+ str(start_pnt[1]))
                    elif self.quadrant == 2:
                        end_pnt = [start_pnt[0], start_pnt[1]+road_dist]
                        road_box = [ [start_pnt[0]-box_margin  , start_pnt[1]] 
                                , [start_pnt[0]+box_margin  , start_pnt[1]] 
                                , [end_pnt[0]-box_margin    , end_pnt[1]] ]
                        new_csv.append(str(start_pnt[0])+', '+ str(start_pnt[1]+road_dist*(j+1)/sample))
                        # print('carpose:', car_pose, 'track : ',i,'  ',self.track[0][i],'  error : ',new_csv[-1])
                    elif self.quadrant == 3:
                        end_pnt = [start_pnt[0]-road_dist, start_pnt[1]]
                        road_box = [ [end_pnt[0]    , end_pnt[1]-box_margin] 
                                , [start_pnt[0]  , start_pnt[1]-box_margin] 
                                , [end_pnt[0]    , end_pnt[1]+box_margin] ]
                        new_csv.append(str(start_pnt[0]-road_dist*(j+1)/sample)+', '+ str(start_pnt[1]))
                        # print('carpose:', car_pose, 'track : ',i,'  ',self.track[0][i],'  error : ',new_csv[-1])
                    elif self.quadrant == 4:
                        end_pnt = [start_pnt[0], start_pnt[1]-road_dist]
                        road_box = [ [end_pnt[0]-box_margin     , end_pnt[1]] 
                                , [end_pnt[0]+box_margin     , end_pnt[1]] 
                                , [start_pnt[0]-box_margin   , start_pnt[1]] ]
                        new_csv.append(str(start_pnt[0])+', '+ str(start_pnt[1]-road_dist*(j+1)/sample))
                        # print(str([start_pnt[0], start_pnt[1]-road_dist*(j+1)/sample]))
                        # print('carpose:', car_pose, 'track : ',i,'  ',self.track[0][i],'  error : ',new_csv[-1])

                last_position = end_pnt
                # print(str(i)+'st  '+str(end_pnt[0])+' '+str(end_pnt[1]))
            
            elif self.track[0][i] == "Left":
                if self.follow_lane == "left":
                    road_dist -= curve_margin
                elif self.follow_lane == "right":
                    road_dist += curve_margin
                else:
                    pass
                for j in range(sample):
                    start_pnt = last_position
                    if self.quadrant == 1:
                        end_pnt = [start_pnt[0]+road_dist, start_pnt[1]+road_dist]
                        center = [start_pnt[0], start_pnt[1]+road_dist]
                        x = start_pnt[0]
                        y = start_pnt[1]
                        a = center[0]
                        b = center[1]
                        r = (math.pi/2) * (j+1)/sample
                        x_ = (x - a) * math.cos(r) - (y-b) * math.sin(r) + a
                        y_ = (x - a) * math.sin(r) + (y-b) * math.cos(r) + b
                        
                        new_csv.append(str(x_)+', '+ str(y_))
                        # print('carpose:', car_pose, 'track : ',i,'  ',self.track[0][i],'  error : ',new_csv[-1])
                        
                    elif self.quadrant == 2:
                        end_pnt = [start_pnt[0]-road_dist, start_pnt[1]+road_dist]
                        center = [start_pnt[0]-road_dist, start_pnt[1]]
                        x = start_pnt[0]
                        y = start_pnt[1]
                        a = center[0]
                        b = center[1]
                        r = (math.pi/2) * (j+1)/sample
                        x_ = (x - a) * math.cos(r) - (y-b) * math.sin(r) + a
                        y_ = (x - a) * math.sin(r) + (y-b) * math.cos(r) + b
                        new_csv.append(str(x_)+', '+ str(y_))
                        # print('carpose:', car_pose, 'track : ',i,'  ',self.track[0][i],'  error : ',new_csv[-1])
                            
                    elif self.quadrant == 3:
                        end_pnt = [start_pnt[0]-road_dist, start_pnt[1]-road_dist]
                        center = [start_pnt[0], start_pnt[1]-road_dist]
                        x = start_pnt[0]
                        y = start_pnt[1]
                        a = center[0]
                        b = center[1]
                        r = (math.pi/2) * (j+1)/sample
                        x_ = (x - a) * math.cos(r) - (y-b) * math.sin(r) + a
                        y_ = (x - a) * math.sin(r) + (y-b) * math.cos(r) + b
                        new_csv.append(str(x_)+', '+ str(y_))
                        # print('carpose:', car_pose, 'track : ',i,'  ',self.track[0][i],'  error : ',new_csv[-1])
                            
                    elif self.quadrant == 4:
                        end_pnt = [start_pnt[0]+road_dist, start_pnt[1]-road_dist]
                        center = [start_pnt[0]+road_dist, start_pnt[1]]
                        x = start_pnt[0]
                        y = start_pnt[1]
                        a = center[0]
                        b = center[1]
                        r = (math.pi/2) * (j+1)/sample
                        x_ = (x - a) * math.cos(r) - (y-b) * math.sin(r) + a
                        y_ = (x - a) * math.sin(r) + (y-b) * math.cos(r) + b
                        new_csv.append(str(x_)+', '+ str(y_))
                        # print('carpose:', car_pose, 'track : ',i,'  ',self.track[0][i],'  error : ',new_csv[-1])
                self.quadrant += 1
                
                if self.quadrant == 0:
                    self.quadrant = 4
                elif self.quadrant == 5:
                    self.quadrant = 1
                last_position = end_pnt
                    # print(str(i)+'st  '+str(end_pnt[0])+' '+str(end_pnt[1]))
                
            elif self.track[0][i] == "Right":
                if self.follow_lane == "left":
                    road_dist += curve_margin
                elif self.follow_lane == "right":
                    road_dist -= curve_margin
                else:
                    pass
                
                for j in range(sample):
                    start_pnt = last_position
                    if self.quadrant == 1:
                        end_pnt = [start_pnt[0]+road_dist, start_pnt[1]-road_dist]
                        center = [start_pnt[0], start_pnt[1]-road_dist]
                        x = start_pnt[0]
                        y = start_pnt[1]
                        a = center[0]
                        b = center[1]
                        r = - (math.pi/2) * (j+1)/sample
                        x_ = (x - a) * math.cos(r) - (y-b) * math.sin(r) + a
                        y_ = (x - a) * math.sin(r) + (y-b) * math.cos(r) + b
                        new_csv.append(str(x_)+', '+ str(y_))
                        # print('carpose:', car_pose, 'track : ',i,'  ',self.track[0][i],'  error : ',new_csv[-1])
                    elif self.quadrant == 2:
                        end_pnt = [start_pnt[0]+road_dist, start_pnt[1]+road_dist]
                        center = [start_pnt[0]+road_dist, start_pnt[1]]
                        x = start_pnt[0]
                        y = start_pnt[1]
                        a = center[0]
                        b = center[1]
                        r = - (math.pi/2) * (j+1)/sample
                        x_ = (x - a) * math.cos(r) - (y-b) * math.sin(r) + a
                        y_ = (x - a) * math.sin(r) + (y-b) * math.cos(r) + b
                        new_csv.append(str(x_)+', '+ str(y_))
                        # print('carpose:', car_pose, 'track : ',i,'  ',self.track[0][i],'  error : ',new_csv[-1])
                    elif self.quadrant == 3:
                        end_pnt = [start_pnt[0]-road_dist, start_pnt[1]+road_dist]
                        center = [start_pnt[0], start_pnt[1]+road_dist]
                        x = start_pnt[0]
                        y = start_pnt[1]
                        a = center[0]
                        b = center[1]
                        r = - (math.pi/2) * (j+1)/sample
                        x_ = (x - a) * math.cos(r) - (y-b) * math.sin(r) + a
                        y_ = (x - a) * math.sin(r) + (y-b) * math.cos(r) + b
                        new_csv.append(str(x_)+', '+ str(y_))
                        # print('carpose:', car_pose, 'track : ',i,'  ',self.track[0][i],'  error : ',new_csv[-1])
                    elif self.quadrant == 4:
                        end_pnt = [start_pnt[0]-road_dist, start_pnt[1]-road_dist]
                        center = [start_pnt[0]-road_dist, start_pnt[1]]
                        x = start_pnt[0]
                        y = start_pnt[1]
                        a = center[0]
                        b = center[1]
                        r = - (math.pi/2) * (j+1)/sample
                        x_ = (x - a) * math.cos(r) - (y-b) * math.sin(r) + a
                        y_ = (x - a) * math.sin(r) + (y-b) * math.cos(r) + b
                        new_csv.append(str(x_)+', '+ str(y_))
                        # print('carpose:', car_pose, 'track : ',i,'  ',self.track[0][i],'  error : ',new_csv[-1])
                self.quadrant -= 1
                # print(str(i)+'st  '+str(end_pnt[0])+' '+str(end_pnt[1]))
                if self.quadrant == 0:
                    self.quadrant = 4
                elif self.quadrant == 5:
                    self.quadrant = 1
                last_position = end_pnt
        
        # write a new csv
        new_csv_fh = open(data_path + "/ground_truth.csv", 'w')
        for i in range(len(new_csv)):
            new_csv_fh.write(new_csv[i]+ '\n')
        new_csv_fh.close()
    
    ###########################################################################
    #
    def _prepare_data(self, csv_path):
        self.data = DriveData(csv_path)
        self.data.read(normalize = False)
    
        self.car_pose = self.data.positions_xyz
        self.car_steering = self.data.measurements
        self.car_velocities = self.data.velocities_xyz
        self.car_velocity = self.data.velocities
        self.car_times = self.data.time_stamps
        self.num_car_pose = len(self.car_pose)
        
        print('Pose samples: {0}'.format(self.num_car_pose))
    
    def _cal_totalerror(self):
        # csv파일에서 x,y 위치를 읽어온뒤 self.car_pose 저장
        # self._cal_roadformula(self.car_pose)
        # 
        self._prepare_data(self.csv_path)
        for i in range(self.num_car_pose):
            # print('',i,'st')
            # print(self.car_pose)
            self._cal_roadformula([self.car_pose[i][0], self.car_pose[i][1]])
        print('mmdc : '  +str(format(self._cal_mmdc(self.total_error), ".9f")))
        print('mddc : ' +str(format(self._cal_mddc(self.total_error_nabs, self.car_times), ".9f")))
        print('emdc : ' +str(format(self._cal_emdc(self.total_error, self.car_times), ".9f")))
        print('mdc  : '  +str(format(self._cal_mdc(self.total_error), ".9f")))
        print('mce  : '  +str(format(self._cal_mce(self.car_steering, self.car_times), ".9f")))
        print('var : ' +str(format(self._cal_var(self.total_error), ".9f")))
        print('dist : ' +str(format(self._cal_dist(self.car_pose), ".9f")))
        print('maxvel : ' +str(format(self._cal_maxvel(self.car_velocity), ".9f")))
        print('avrvel : ' +str(format(self._cal_avrvel(self.car_velocity), ".9f")))
        self._cal_ggdiagram(self.car_velocities, self.car_times)
        error_txt=[]
        error_txt.append('mmdc , ' +str(format(self._cal_mmdc(self.total_error), ".9f")))
        error_txt.append('mddc , ' +str(format(self._cal_mddc(self.total_error_nabs, self.car_times), ".9f")))
        error_txt.append('emdc , ' +str(format(self._cal_emdc(self.total_error, self.car_times), ".9f")))
        error_txt.append('mdc  , ' +str(format(self._cal_mdc(self.total_error), ".9f")))
        error_txt.append('mce  , ' +str(format(self._cal_mce(self.car_steering, self.car_times), ".9f")))
        error_txt.append('var  , ' +str(format(self._cal_var(self.total_error), ".9f")))
        error_txt.append('dist : ' +str(format(self._cal_dist(self.car_pose), ".9f")))
        error_txt.append('maxvel : ' +str(format(self._cal_maxvel(self.car_velocity), ".9f")))
        error_txt.append('avrvel : ' +str(format(self._cal_avrvel(self.car_velocity), ".9f")))
        # self._build_csv(self.csv_path[:-len(self.csv_path.split('/')[-1])], self.total_error_nabs)
        self._build_csv(self.csv_path[:-len(self.csv_path.split('/')[-1])], error_txt)
        self._build_txt(self.csv_path[:-len(self.csv_path.split('/')[-1])], error_txt)
        # print('total error : ',self.total_error)
        
        
    def _cal_roadformula(self, car_pose):
        # print(car_pose[0], car_pose[1])
        last_position = [0, 0]
        for i in range(0, len(self.track[0])):
            road_dist = float(self.track[1][i])
            curve_margin = 2.333
            road_box = [0, 0, 0]
            box_margin = 5.0
            # box_margin = 50.0
            if self.track[0][i] == "Straight":
                start_pnt = last_position
                if self.quadrant == 1:
                    end_pnt = [start_pnt[0]+road_dist, start_pnt[1]]
                    road_box = [ [start_pnt[0]  , start_pnt[1]-box_margin] 
                               , [end_pnt[0]    , end_pnt[1]-box_margin] 
                               , [start_pnt[0]  , start_pnt[1]+box_margin] ]
                    if road_box[0][0] < car_pose[0] < road_box[1][0] and road_box[0][1] < car_pose[1] < road_box[2][1]:
                        self.total_error.append(abs(start_pnt[1] - car_pose[1]))
                        self.total_error_nabs.append((start_pnt[1] - car_pose[1]))
                        # print('carpose:', car_pose, 'track : ',i,'  ',self.track[0][i],'  error : ',self.total_error[-1])
                elif self.quadrant == 2:
                    end_pnt = [start_pnt[0], start_pnt[1]+road_dist]
                    road_box = [ [start_pnt[0]-box_margin  , start_pnt[1]] 
                               , [start_pnt[0]+box_margin  , start_pnt[1]] 
                               , [end_pnt[0]-box_margin    , end_pnt[1]] ]
                    if road_box[0][0] < car_pose[0] < road_box[1][0] and road_box[0][1] < car_pose[1] < road_box[2][1]:
                        self.total_error.append(abs(start_pnt[0] - car_pose[0]))
                        self.total_error_nabs.append((start_pnt[0] - car_pose[0]))
                        # print('carpose:', car_pose, 'track : ',i,'  ',self.track[0][i],'  error : ',self.total_error[-1])
                elif self.quadrant == 3:
                    end_pnt = [start_pnt[0]-road_dist, start_pnt[1]]
                    road_box = [ [end_pnt[0]    , end_pnt[1]-box_margin] 
                               , [start_pnt[0]  , start_pnt[1]-box_margin] 
                               , [end_pnt[0]    , end_pnt[1]+box_margin] ]
                    if road_box[0][0] < car_pose[0] < road_box[1][0] and road_box[0][1] < car_pose[1] < road_box[2][1]:
                        self.total_error.append(abs(start_pnt[1] - car_pose[1]))
                        self.total_error_nabs.append((start_pnt[1] - car_pose[1]))
                        # print('carpose:', car_pose, 'track : ',i,'  ',self.track[0][i],'  error : ',self.total_error[-1])
                elif self.quadrant == 4:
                    end_pnt = [start_pnt[0], start_pnt[1]-road_dist]
                    road_box = [ [end_pnt[0]-box_margin     , end_pnt[1]] 
                               , [end_pnt[0]+box_margin     , end_pnt[1]] 
                               , [start_pnt[0]-box_margin   , start_pnt[1]] ]
                    if road_box[0][0] < car_pose[0] < road_box[1][0] and road_box[0][1] < car_pose[1] < road_box[2][1]:
                        self.total_error.append(abs(start_pnt[0] - car_pose[0]))
                        self.total_error_nabs.append((start_pnt[0] - car_pose[0]))
                        # print('carpose:', car_pose, 'track : ',i,'  ',self.track[0][i],'  error : ',self.total_error[-1])

                last_position = end_pnt
                # print(str(i)+'st  '+str(end_pnt[0])+' '+str(end_pnt[1]))
            
            elif self.track[0][i] == "Left":
                if self.follow_lane == "left":
                    road_dist -= curve_margin
                elif self.follow_lane == "right":
                    road_dist += curve_margin
                else:
                    pass
                
                start_pnt = last_position
                if self.quadrant == 1:
                    end_pnt = [start_pnt[0]+road_dist, start_pnt[1]+road_dist]
                    center = [start_pnt[0], start_pnt[1]+road_dist]
                    if start_pnt[0] < car_pose[0] < end_pnt[0]+box_margin and start_pnt[1]-box_margin < car_pose[1] < end_pnt[1]:
                        dx = car_pose[0]-center[0]
                        dy = car_pose[1]-center[1]
                        self.total_error.append(abs(math.sqrt( (dx**2) + (dy**2) ) - road_dist))
                        self.total_error_nabs.append((math.sqrt( (dx**2) + (dy**2) ) - road_dist))
                        # print('carpose:', car_pose, 'track : ',i,'  ',self.track[0][i],'  error : ',self.total_error[-1])
                    
                elif self.quadrant == 2:
                    end_pnt = [start_pnt[0]-road_dist, start_pnt[1]+road_dist]
                    center = [start_pnt[0]-road_dist, start_pnt[1]]
                    if end_pnt[0] < car_pose[0] < start_pnt[0]+box_margin and start_pnt[1] < car_pose[1] < end_pnt[1]+box_margin:
                        dx = car_pose[0]-center[0]
                        dy = car_pose[1]-center[1]
                        self.total_error.append(abs(math.sqrt( (dx**2) + (dy**2) ) - road_dist))
                        self.total_error_nabs.append((math.sqrt( (dx**2) + (dy**2) ) - road_dist))
                        # print('carpose:', car_pose, 'track : ',i,'  ',self.track[0][i],'  error : ',self.total_error[-1])
                        
                elif self.quadrant == 3:
                    end_pnt = [start_pnt[0]-road_dist, start_pnt[1]-road_dist]
                    center = [start_pnt[0], start_pnt[1]-road_dist]
                    if end_pnt[0]-box_margin < car_pose[0] < start_pnt[0] and end_pnt[1] < car_pose[1] < start_pnt[1]+box_margin:
                        dx = car_pose[0]-center[0]
                        dy = car_pose[1]-center[1]
                        self.total_error.append(abs(math.sqrt( (dx**2) + (dy**2) ) - road_dist))
                        self.total_error_nabs.append((math.sqrt( (dx**2) + (dy**2) ) - road_dist))
                        # print('carpose:', car_pose, 'track : ',i,'  ',self.track[0][i],'  error : ',self.total_error[-1])
                        
                elif self.quadrant == 4:
                    end_pnt = [start_pnt[0]+road_dist, start_pnt[1]-road_dist]
                    center = [start_pnt[0]+road_dist, start_pnt[1]]
                    if start_pnt[0]-box_margin < car_pose[0] < end_pnt[0] and end_pnt[1]-box_margin < car_pose[1] < start_pnt[1]:
                        dx = car_pose[0]-center[0]
                        dy = car_pose[1]-center[1]
                        self.total_error.append(abs(math.sqrt( (dx**2) + (dy**2) ) - road_dist))
                        self.total_error_nabs.append((math.sqrt( (dx**2) + (dy**2) ) - road_dist))
                        # print('carpose:', car_pose, 'track : ',i,'  ',self.track[0][i],'  error : ',self.total_error[-1])
                last_position = end_pnt
                self.quadrant += 1
                # print(str(i)+'st  '+str(end_pnt[0])+' '+str(end_pnt[1]))
                
            elif self.track[0][i] == "Right":
                if self.follow_lane == "left":
                    road_dist += curve_margin
                elif self.follow_lane == "right":
                    road_dist -= curve_margin
                else:
                    pass
                
                start_pnt = last_position
                if self.quadrant == 1:
                    end_pnt = [start_pnt[0]+road_dist, start_pnt[1]-road_dist]
                    center = [start_pnt[0], start_pnt[1]-road_dist]
                    if start_pnt[0] < car_pose[0] < end_pnt[0]+box_margin and end_pnt[1] < car_pose[1] < start_pnt[1]+box_margin:
                        dx = car_pose[0]-center[0]
                        dy = car_pose[1]-center[1]
                        self.total_error.append(abs(math.sqrt( (dx**2) + (dy**2) ) - road_dist))
                        self.total_error_nabs.append((math.sqrt( (dx**2) + (dy**2) ) - road_dist))
                        # print('carpose:', car_pose, 'track : ',i,'  ',self.track[0][i],'  error : ',self.total_error[-1])
                elif self.quadrant == 2:
                    end_pnt = [start_pnt[0]+road_dist, start_pnt[1]+road_dist]
                    center = [start_pnt[0]+road_dist, start_pnt[1]]
                    if start_pnt[0]-box_margin < car_pose[0] < end_pnt[0] and start_pnt[1] < car_pose[1] < end_pnt[1]+box_margin:
                        dx = car_pose[0]-center[0]
                        dy = car_pose[1]-center[1]
                        self.total_error.append(abs(math.sqrt( (dx**2) + (dy**2) ) - road_dist))
                        self.total_error_nabs.append((math.sqrt( (dx**2) + (dy**2) ) - road_dist))
                        # print('carpose:', car_pose, 'track : ',i,'  ',self.track[0][i],'  error : ',self.total_error[-1])
                elif self.quadrant == 3:
                    end_pnt = [start_pnt[0]-road_dist, start_pnt[1]+road_dist]
                    center = [start_pnt[0], start_pnt[1]+road_dist]
                    if end_pnt[0]-box_margin < car_pose[0] < start_pnt[0] and start_pnt[1]-box_margin < car_pose[1] < end_pnt[1]:
                        dx = car_pose[0]-center[0]
                        dy = car_pose[1]-center[1]
                        self.total_error.append(abs(math.sqrt( (dx**2) + (dy**2) ) - road_dist))
                        self.total_error_nabs.append((math.sqrt( (dx**2) + (dy**2) ) - road_dist))
                        # print('carpose:', car_pose, 'track : ',i,'  ',self.track[0][i],'  error : ',self.total_error[-1])
                elif self.quadrant == 4:
                    end_pnt = [start_pnt[0]-road_dist, start_pnt[1]-road_dist]
                    center = [start_pnt[0]-road_dist, start_pnt[1]]
                    if end_pnt[0] < car_pose[0] < start_pnt[0]+box_margin and end_pnt[1]-box_margin < car_pose[1] < start_pnt[1]:
                        dx = car_pose[0]-center[0]
                        dy = car_pose[1]-center[1]
                        self.total_error.append(abs(math.sqrt( (dx**2) + (dy**2) ) - road_dist))
                        self.total_error_nabs.append((math.sqrt( (dx**2) + (dy**2) ) - road_dist))
                        # print('carpose:', car_pose, 'track : ',i,'  ',self.track[0][i],'  error : ',self.total_error[-1])
                last_position = end_pnt
                self.quadrant -= 1
                # print(str(i)+'st  '+str(end_pnt[0])+' '+str(end_pnt[1]))
            # print(str(i)+'st  '+str(end_pnt[0])+' '+str(end_pnt[1]))
                
            if self.quadrant == 0:
                self.quadrant = 4
            elif self.quadrant == 5:
                self.quadrant = 1
    
    def _cal_var(self, error):
        variance = np.array(error)
        variance = np.var(variance)
        return variance
    
    def _cal_dist(self, dist):
        total_dist = 0
        num_data = len(dist) - 1
        for i in range(num_data):
            distance_x = float(dist[i+1][0]) - float(dist[i][0])
            distance_y = float(dist[i+1][1]) - float(dist[i][1])
            distance_z = float(dist[i+1][2]) - float(dist[i][2])
            
            total_dist += math.sqrt(distance_x**2 + distance_y**2 + distance_z**2)
            
        return total_dist/4
    
    def _cal_avrvel(self, vel):
        avr_vel = 0
        num_data = len(vel)
        for i in range(num_data):
            avr_vel += vel[i]
        avr_vel /= num_data
        return avr_vel
    
    def _cal_maxvel(self, vel):
        max_vel = 0
        num_data = len(vel)
        for i in range(num_data):
            max_vel = max_vel if max_vel > vel[i] else vel[i]
        return max_vel
    
    def _cal_emdc(self, error, t):
        mmdc = self._cal_mmdc(error)
        mddc = self._cal_mddc(error, t)
        a = 0.5
        return (1-a)*mmdc + a*mddc
    
    def _cal_mmdc(self, error):
        error_mmdc = 0
        num_data = len(error)
        for i in range(num_data):
            if abs(error[i]) > 1.15:
                error_mmdc += abs(error[i])
            elif abs(error[i]) < 0.7071:
                error_mmdc += 0
            else:
                error_mmdc += 5.256 * ( 0.2*(error[i]**4) - 0.1*(error[i]**2) )
        error_mmdc /= num_data
        return error_mmdc
    
    def _cal_mddc(self, error, t):
        error_mddc = 0
        num_data = len(error) - 1
        # print(len(error))
        # print(len(t))
        count = 0
        for i in range(num_data):
            de = abs(error[i+1] - error[i])
            dt = t[i+1] - t[i]
            
            if dt > 1.0 or dt < 0:
                count += 1
            else :
                error_mddc += de/dt
            
        error_mddc /= (num_data - count)
        return error_mddc
    
    def _cal_mdc(self, error):
        error_mdc = 0
        num_data = len(error)
        for i in range(num_data):
            error_mdc += error[i]
        error_mdc /= num_data
        return error_mdc
    
    
    def _cal_mce(self, steering, time):
        error_mce = 0
        num_data = len(steering) - 1
        count = 0
        for i in range(num_data):
            time_diff = float(time[i+1]) - float(time[i])
            if time_diff > 1.0 or time_diff < 0:
                count += 1
            else :
                error_mce += (steering[i+1][0] - steering[i][0])**2
            
        error_mce /= (num_data - count)
        return error_mce
    
    def _cal_ggdiagram(self, v, t):
        import matplotlib.pyplot as plt
        import pandas as pd
        import numpy as np
        
        ax = []
        ay = []
        av_ax_g = []
        av_ay_g = []
        num_data = len(v) - 1
        av_num = 20 #Simple Moving Average
        count = 0
        for i in range(num_data):
            dvx = v[i+1][0] - v[i][0]
            dvy = v[i+1][1] - v[i][1]
            dt = t[i+1] - t[i]
            if dt > 1.0 or dt < 0:
                count += 1
            else :
                ax.append(dvx/dt)
                ay.append(dvy/dt)
                if len(ax) is av_num:
                    av_ax = sum(ax) / av_num
                    av_ay = sum(ay) / av_num
                    av_ax_g.append(av_ax/9.8)
                    av_ay_g.append(av_ay/9.8)
                    del ax[0], ay[0]
           
        ##########local frame vel###########
        plt.rcParams["figure.figsize"] = (10,4)
        plt.rcParams['axes.grid'] = True
        plt.figure()
        plt.title('Ax-t diagram', fontsize=20)
        plt.xlabel('t', fontsize=14)
        plt.ylabel('Ax(g)', fontsize=14)
        plt.ylim([-1.5, 1.5])
        # plt.scatter(range(0, len(ax[:-1])), ax[:-1], s=.1)
        plt.plot(range(0, len(av_ax_g)), av_ax_g,'-', color = 'red', markersize=1)
        self._savefigs(plt, self.csv_path[:-len(self.csv_path.split('/')[-1])]+'Ax-t_Diagram')
        
        plt.rcParams["figure.figsize"] = (10,4)
        plt.rcParams['axes.grid'] = True
        plt.figure()
        plt.title('Ay-t diagram', fontsize=20)
        plt.xlabel('t', fontsize=14)
        plt.ylabel('Ay(g)', fontsize=14)
        plt.ylim([-1.5, 1.5])
        # plt.scatter(range(0, len(av_ay)), av_ay, s=.1)
        plt.plot(range(0, len(av_ay_g)), av_ay_g,'-', color = 'blue', markersize=1)
        self._savefigs(plt, self.csv_path[:-len(self.csv_path.split('/')[-1])]+'Ay-t_Diagram')
        
        # print(ax, ay)
        plt.rcParams['axes.grid'] = True
        plt.figure()
        # plt.plot(ax[0], ay[1])
        plt.axis('equal')
        _, axes = plt.subplots()
        c_center = (0,0)
        c_radius = 1
        draw_circle = plt.Circle(c_center, c_radius, fc='w', ec='r', fill=False, linestyle='--')
        axes.set_aspect(1)
        axes.add_artist(draw_circle)
        
        plt.title('G-G diagram', fontsize=20)
        plt.xlabel('Ax(g)', fontsize=14)
        plt.ylabel('Ay(g)', fontsize=14)
        plt.xlim([-1.5, 1.5])
        plt.ylim([-1.5, 1.5])
        plt.scatter(av_ax_g[500:1500], av_ay_g[500:1500], s=.1)
        self._savefigs(plt, self.csv_path[:-len(self.csv_path.split('/')[-1])]+'G-G_Diagram')
        
    
    def _savefigs(self, plt, filename):
        plt.savefig(filename + '.png', dpi=150)
        # plt.savefig(filename + '.pdf', dpi=150)
        print('Saved ' + filename + '.png & .pdf.')
        
def main(csv_path, follow_lane):
    rospy.init_node('mapinfo_generator')
    m = MapInfoGenerator(csv_path, follow_lane)
    rospy.spin()
        
        
if __name__ == '__main__':
    try:
        if (len(sys.argv) != 3):
            exit('Usage:\n$ rosrun mapinfo_generator {} csv_path baseline(right, left, center)'.format(sys.argv[0]))

        main(sys.argv[1], sys.argv[2])

    except KeyboardInterrupt:
        print ('\nShutdown requested. Exiting...')
