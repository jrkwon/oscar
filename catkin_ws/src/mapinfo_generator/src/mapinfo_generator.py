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

        print('mdc  : '  +str(format(self._cal_mdc(self.total_error), ".9f")))
        print('emdc : '  +str(format(self._cal_emdc(self.total_error), ".9f")))
        print('mce  : '  +str(format(self._cal_mce(self.car_steering), ".9f")))
        print('mddc : ' +str(format(self._cal_mddc(self.total_error, self.car_times), ".9f")))
        print('emddc : ' +str(format(self._cal_emddc(self.total_error, self.car_times), ".9f")))
        print('var : ' +str(format(self._cal_var(self.total_error), ".9f")))
        self._cal_ggdiagram(self.car_velocities, self.car_times)
        error_txt=[]
        error_txt.append('mdc  : ' +str(format(self._cal_mdc(self.total_error), ".9f")))
        error_txt.append('emdc : ' +str(format(self._cal_emdc(self.total_error), ".9f")))
        error_txt.append('mce  : ' +str(format(self._cal_mce(self.car_steering), ".9f")))
        error_txt.append('mddc : ' +str(format(self._cal_mddc(self.total_error, self.car_times), ".9f")))
        error_txt.append('emddc : ' +str(format(self._cal_emddc(self.total_error, self.car_times), ".9f")))
        error_txt.append('var  : ' +str(format(self._cal_var(self.total_error), ".9f")))
        
        self._build_txt(self.csv_path[:-len(self.csv_path.split('/')[-1])], error_txt)
        # print('total error : ',self.total_error)
        
        
    def _cal_roadformula(self, car_pose):
        # print(car_pose[0], car_pose[1])
        last_position = [0, 0]
        for i in range(0, len(self.track[0])):
            road_dist = float(self.track[1][i])
            curve_margin = 2.333
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
    
    def _cal_var(self, error):
        variance = np.array(error)
        variance = np.var(variance)
        return variance
    
    def _cal_emddc(self, error, t):
        emdc = self._cal_emdc(error)
        mddc = self._cal_mddc(error, t)
        a = 0.5
        return (1-a)*emdc + a*mddc
    
    def _cal_emdc(self, error):
        error_emdc = 0
        num_data = len(error)
        for i in range(num_data):
            if abs(error[i]) > 1.15:
                error_emdc += error[i]
            elif abs(error[i]) < 0.7071:
                error_emdc += 0
            else:
                error_emdc += 5.256 * ( 0.2*(error[i]**4) - 0.1*(error[i]**2) )
        error_emdc /= num_data
        return error_emdc
    
    def _cal_mddc(self, error, t):
        error_mddc = 0
        num_data = len(error) - 1
        for i in range(num_data):
            de = abs(error[i+1] - error[i])
            dt = t[i+1] - t[i]
            error_mddc += de/dt
            
        error_mddc /= num_data
        return error_mddc
    
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
        
        ax = []
        ay = []
        av_ax_g = []
        av_ay_g = []
        num_data = len(v) - 1
        av_num = 20 #Simple Moving Average
        for i in range(num_data):
            dvx = v[i+1][0] - v[i][0]
            dvy = v[i+1][1] - v[i][1]
            dt = t[i+1] - t[i]
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
            exit('Usage:\n$ rosrun mapinfo_generator {} ground_truth_path baseline(right, left, center)'.format(sys.argv[0]))

        main(sys.argv[1], sys.argv[2])

    except KeyboardInterrupt:
        print ('\nShutdown requested. Exiting...')
