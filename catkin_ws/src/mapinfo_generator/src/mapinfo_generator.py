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
from drive_data import DriveData
from scipy import interpolate

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
        # self._cal_groundtruth(self.oscar_path+"/neural_net/map_info")
        # os.system("rosnode kill " + "mapinfo_generator")
        
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
                    
                    for n in range(len(self.last_position)):
                        if self.last_position[n] < 0.0001 and self.last_position[n] > -0.0001:
                            self.last_position[n] = 0
                    new_csv.append(str(self.last_position[0])+" "+str(self.last_position[1])+'\n')
                
            elif self.track[0][i] == "Left" or self.track[0][i] == "Right":
                road_dist = float(self.track[1][i]) #도로의 길이 (m). 
                data_term = 1                       #1cm 마다 pose를 구할것임
                theta = math.pi / (2 * (road_dist/data_term))
                
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
    
    ###########################################################################
    #
    def _prepare_data(self, csv_path):
        self.data = DriveData(csv_path)
        self.data.read(normalize = False)
    
        self.car_pose = self.data.positions_xyz
        self.car_steering = self.data.measurements
        self.car_velocities = self.data.velocities_xyz
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

        print('mdc : '+str(self._cal_mdc(self.total_error)))
        print('mce : '+str(format(self._cal_mce(self.car_steering), ".9f")))
        self._cal_ggdiagram(self.car_velocities, self.car_times)
        # print('total error : ',self.total_error)
        
        
    def _cal_roadformula(self, car_pose):
        # print(car_pose[0], car_pose[1])
        last_position = [0, 0]
        for i in range(0, len(self.track[0])):
            road_dist = float(self.track[1][i])
            curve_margin = 2.3
            road_box = [0, 0, 0]
            box_margin = 50.0
            if self.track[0][i] == "Straight":
                start_pnt = last_position
                if self.quadrant == 1:
                    end_pnt = [start_pnt[0]+road_dist, start_pnt[1]]
                    road_box = [ [start_pnt[0]  , start_pnt[1]-box_margin] 
                               , [end_pnt[0]    , end_pnt[1]-box_margin] 
                               , [start_pnt[0]  , start_pnt[1]+box_margin] ]
                    if road_box[0][0] < car_pose[0] < road_box[1][0] and road_box[0][1] < car_pose[1] < road_box[2][1]:
                        self.total_error.append(abs(start_pnt[1] - car_pose[1]))
                        # print('carpose:', car_pose, 'track : ',i,'  ',self.track[0][i],'  error : ',self.total_error[-1])
                elif self.quadrant == 2:
                    end_pnt = [start_pnt[0], start_pnt[1]+road_dist]
                    road_box = [ [start_pnt[0]-box_margin  , start_pnt[1]] 
                               , [start_pnt[0]+box_margin  , start_pnt[1]] 
                               , [end_pnt[0]-box_margin    , end_pnt[1]] ]
                    if road_box[0][0] < car_pose[0] < road_box[1][0] and road_box[0][1] < car_pose[1] < road_box[2][1]:
                        self.total_error.append(abs(start_pnt[0] - car_pose[0]))
                        # print('carpose:', car_pose, 'track : ',i,'  ',self.track[0][i],'  error : ',self.total_error[-1])
                elif self.quadrant == 3:
                    end_pnt = [start_pnt[0]-road_dist, start_pnt[1]]
                    road_box = [ [end_pnt[0]    , end_pnt[1]-box_margin] 
                               , [start_pnt[0]  , start_pnt[1]-box_margin] 
                               , [end_pnt[0]    , end_pnt[1]+box_margin] ]
                    if road_box[0][0] < car_pose[0] < road_box[1][0] and road_box[0][1] < car_pose[1] < road_box[2][1]:
                        self.total_error.append(abs(start_pnt[1] - car_pose[1]))
                        # print('carpose:', car_pose, 'track : ',i,'  ',self.track[0][i],'  error : ',self.total_error[-1])
                elif self.quadrant == 4:
                    end_pnt = [start_pnt[0], start_pnt[1]-road_dist]
                    road_box = [ [end_pnt[0]-box_margin     , end_pnt[1]] 
                               , [end_pnt[0]+box_margin     , end_pnt[1]] 
                               , [start_pnt[0]-box_margin   , start_pnt[1]] ]
                    if road_box[0][0] < car_pose[0] < road_box[1][0] and road_box[0][1] < car_pose[1] < road_box[2][1]:
                        self.total_error.append(abs(start_pnt[0] - car_pose[0]))
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
                        # print('carpose:', car_pose, 'track : ',i,'  ',self.track[0][i],'  error : ',self.total_error[-1])
                    
                elif self.quadrant == 2:
                    end_pnt = [start_pnt[0]-road_dist, start_pnt[1]+road_dist]
                    center = [start_pnt[0]-road_dist, start_pnt[1]]
                    if end_pnt[0] < car_pose[0] < start_pnt[0]+box_margin and start_pnt[1] < car_pose[1] < end_pnt[1]+box_margin:
                        dx = car_pose[0]-center[0]
                        dy = car_pose[1]-center[1]
                        self.total_error.append(abs(math.sqrt( (dx**2) + (dy**2) ) - road_dist))
                        # print('carpose:', car_pose, 'track : ',i,'  ',self.track[0][i],'  error : ',self.total_error[-1])
                        
                elif self.quadrant == 3:
                    end_pnt = [start_pnt[0]-road_dist, start_pnt[1]-road_dist]
                    center = [start_pnt[0], start_pnt[1]-road_dist]
                    if end_pnt[0]-box_margin < car_pose[0] < start_pnt[0] and end_pnt[1] < car_pose[1] < start_pnt[1]+box_margin:
                        dx = car_pose[0]-center[0]
                        dy = car_pose[1]-center[1]
                        self.total_error.append(abs(math.sqrt( (dx**2) + (dy**2) ) - road_dist))
                        # print('carpose:', car_pose, 'track : ',i,'  ',self.track[0][i],'  error : ',self.total_error[-1])
                        
                elif self.quadrant == 4:
                    end_pnt = [start_pnt[0]+road_dist, start_pnt[1]-road_dist]
                    center = [start_pnt[0]+road_dist, start_pnt[1]]
                    if start_pnt[0]-box_margin < car_pose[0] < end_pnt[0] and end_pnt[1]-box_margin < car_pose[1] < start_pnt[1]:
                        dx = car_pose[0]-center[0]
                        dy = car_pose[1]-center[1]
                        self.total_error.append(abs(math.sqrt( (dx**2) + (dy**2) ) - road_dist))
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
                        # print('carpose:', car_pose, 'track : ',i,'  ',self.track[0][i],'  error : ',self.total_error[-1])
                elif self.quadrant == 2:
                    end_pnt = [start_pnt[0]+road_dist, start_pnt[1]+road_dist]
                    center = [start_pnt[0]+road_dist, start_pnt[1]]
                    if start_pnt[0]-box_margin < car_pose[0] < end_pnt[0] and start_pnt[1] < car_pose[1] < end_pnt[1]+box_margin:
                        dx = car_pose[0]-center[0]
                        dy = car_pose[1]-center[1]
                        self.total_error.append(abs(math.sqrt( (dx**2) + (dy**2) ) - road_dist))
                        # print('carpose:', car_pose, 'track : ',i,'  ',self.track[0][i],'  error : ',self.total_error[-1])
                elif self.quadrant == 3:
                    end_pnt = [start_pnt[0]-road_dist, start_pnt[1]+road_dist]
                    center = [start_pnt[0], start_pnt[1]+road_dist]
                    if end_pnt[0]-box_margin < car_pose[0] < start_pnt[0] and start_pnt[1]-box_margin < car_pose[1] < end_pnt[1]:
                        dx = car_pose[0]-center[0]
                        dy = car_pose[1]-center[1]
                        self.total_error.append(abs(math.sqrt( (dx**2) + (dy**2) ) - road_dist))
                        # print('carpose:', car_pose, 'track : ',i,'  ',self.track[0][i],'  error : ',self.total_error[-1])
                elif self.quadrant == 4:
                    end_pnt = [start_pnt[0]-road_dist, start_pnt[1]-road_dist]
                    center = [start_pnt[0]-road_dist, start_pnt[1]]
                    if end_pnt[0] < car_pose[0] < start_pnt[0]+box_margin and end_pnt[1]-box_margin < car_pose[1] < start_pnt[1]:
                        dx = car_pose[0]-center[0]
                        dy = car_pose[1]-center[1]
                        self.total_error.append(abs(math.sqrt( (dx**2) + (dy**2) ) - road_dist))
                        # print('carpose:', car_pose, 'track : ',i,'  ',self.track[0][i],'  error : ',self.total_error[-1])
                last_position = end_pnt
                self.quadrant -= 1
                # print(str(i)+'st  '+str(end_pnt[0])+' '+str(end_pnt[1]))
                
            if self.quadrant == 0:
                self.quadrant = 4
            elif self.quadrant == 5:
                self.quadrant = 1
            
                        
    def _cal_mdc(self, error):
        error_mdc = 0
        num_data = len(error)
        for i in range(num_data):
            error_mdc += error[i]
        error_mdc /= num_data
        return error_mdc
    
    
    def _cal_mce(self, steering):
        error_mce = 0
        num_data = len(steering) - 1
        for i in range(num_data):
            error_mce += (steering[i+1][0] - steering[i][0])**2
            
        error_mce /= num_data
        return error_mce
    
    def _cal_ggdiagram(self, v, t):
        import matplotlib.pyplot as plt
        import pandas as pd
        import numpy as np
        
        # ax = []
        # ay = []
        av_ax=[]
        av_ay=[]
        # ax_b = []
        # ay_b = []
        # 가속도 계산
        num_data = len(v) - 1
        data_count = 0
        av_ax_data = []
        av_ay_data = []
        av_num = 20
        for i in range(num_data):
            # ax.append(v[i][3]/9.8)
            # ay.append(v[i][4])
            dvx = v[i+1][0] - v[i][0]
            dvy = v[i+1][1] - v[i][1]
            # dvx_b = v[i+1][3] - v[i][3]
            # dvy_b = v[i+1][4] - v[i][4]
            dt = t[i+1] - t[i]
            # if dt is not 0:
            #     ax.append((dvx/dt)/9.8)
            #     ay.append(v[i][4])
            #     # ay.append((dvy/dt)/9.8)
            #     # ax_b.append((dvx_b/dt)/9.8)
            #     # ay_b.append((dvy_b/dt)/9.8)
            #     data_count += 1
            #     print("ax : "+str(dvx/dt)+"  ay : "+str(dvy/dt))
            
            # av_ax_data.append(v[i][3])
            av_ax_data.append(dvx/dt)
            av_ay_data.append(dvy/dt)
            if len(av_ax_data) is av_num:
                ax = sum(av_ax_data) / av_num
                ay = sum(av_ay_data) / av_num
                av_ax.append(ax/9.8)
                av_ay.append(ay/9.8)
                # ay.append(v[i][4])
                data_count += 1
                del av_ax_data[0]
                del av_ay_data[0]
           
        ##########local frame vel###########
        plt.rcParams["figure.figsize"] = (10,4)
        plt.rcParams['axes.grid'] = True
        plt.figure()
        plt.title('Ax-t diagram', fontsize=20)
        plt.xlabel('t', fontsize=14)
        plt.ylabel('Ax(g)', fontsize=14)
        plt.ylim([-1.5, 1.5])
        # plt.scatter(range(0, len(ax[:-1])), ax[:-1], s=.1)
        plt.plot(range(0, len(av_ax)), av_ax,'-', color = 'red', markersize=1)
        self._savefigs(plt, self.csv_path[:-len(self.csv_path.split('/')[-1])]+'Ax-t_Diagram')
        
        plt.rcParams["figure.figsize"] = (10,4)
        plt.rcParams['axes.grid'] = True
        plt.figure()
        plt.title('Ay-t diagram', fontsize=20)
        plt.xlabel('t', fontsize=14)
        plt.ylabel('Ay(g)', fontsize=14)
        plt.ylim([-1.5, 1.5])
        # plt.scatter(range(0, len(av_ay)), av_ay, s=.1)
        plt.plot(range(0, len(av_ay)), av_ay,'-', color = 'blue', markersize=1)
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
        plt.scatter(av_ax[500:1500], av_ay[500:1500], s=.1)
        self._savefigs(plt, self.csv_path[:-len(self.csv_path.split('/')[-1])]+'G-G_Diagram')
        
        # ##########body frame vel############
        
        # plt.rcParams["figure.figsize"] = (10,4)
        # plt.rcParams['axes.grid'] = True
        # plt.figure()
        # plt.title('Ax_b-t diagram', fontsize=20)
        # plt.xlabel('t', fontsize=14)
        # plt.ylabel('Ax_b(g)', fontsize=14)
        # plt.ylim([-1.5, 1.5])
        # plt.plot(range(0, len(ax[674:1168])), ax_b[674:1168])
        # self._savefigs(plt, self.csv_path[:-len(self.csv_path.split('/')[-1])]+'Ax_b-t_Diagram')
        
        # plt.rcParams["figure.figsize"] = (10,4)
        # plt.rcParams['axes.grid'] = True
        # plt.figure()
        # plt.title('Ay_b-t diagram', fontsize=20)
        # plt.xlabel('t', fontsize=14)
        # plt.ylabel('Ay_b(g)', fontsize=14)
        # plt.ylim([-1.5, 1.5])
        # plt.plot(range(0, len(ay[674:1168])), ay_b[674:1168])
        # self._savefigs(plt, self.csv_path[:-len(self.csv_path.split('/')[-1])]+'Ay_b-t_Diagram')
        
        # # print(ax, ay)
        # plt.rcParams['axes.grid'] = True
        # plt.figure()
        # # plt.plot(ax[0], ay[1])
        # plt.axis('equal')
        # _, axes = plt.subplots()
        # c_center = (0,0)
        # c_radius = 1
        # draw_circle = plt.Circle(c_center, c_radius, fc='w', ec='r', fill=False, linestyle='--')
        # axes.set_aspect(1)
        # axes.add_artist(draw_circle)
        
        # plt.title('G-G_b diagram', fontsize=20)
        # plt.xlabel('A_b(g)', fontsize=14)
        # plt.ylabel('A_b(g)', fontsize=14)
        # plt.xlim([-1.5, 1.5])
        # plt.ylim([-1.5, 1.5])
        # plt.scatter(ax_b[674:1168], ay_b[674:1168], s=.1)
        # self._savefigs(plt, self.csv_path[:-len(self.csv_path.split('/')[-1])]+'G-G_b_Diagram')
    
    def _savefigs(self, plt, filename):
        plt.savefig(filename + '.png', dpi=150)
        plt.savefig(filename + '.pdf', dpi=150)
        print('Saved ' + filename + '.png & .pdf.')
        
def main(csv_path, follow_lane):
    rospy.init_node('mapinfo_generator')
    m = MapInfoGenerator(csv_path, follow_lane)
    rospy.spin()
        
        
if __name__ == '__main__':
    try:
        if (len(sys.argv) != 3):
            exit('Usage:\n$ rosrun mapinfo_generator {} data_path baseline(right, left, center)'.format(sys.argv[0]))

        main(sys.argv[1], sys.argv[2])

    except KeyboardInterrupt:
        print ('\nShutdown requested. Exiting...')
