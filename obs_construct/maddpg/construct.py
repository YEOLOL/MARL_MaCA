import torch
import numpy as np
import random
import copy
import math
import torch.nn as nn
import torch.nn.functional as F


class ObsConstruct:
    def __init__(self, size_x, size_y, detector_num, fighter_num):
        self.battlefield_size_x = size_x  # 长度
        self.battlefield_size_y = size_y  # 宽度
        self.detector_num = detector_num
        self.fighter_num = fighter_num


    def obs_construct(self, side1_obs_raw_dict):
        fighter_data_obs_list = side1_obs_raw_dict['fighter_obs_list']
        actor_obs_list = self.actor_input(fighter_data_obs_list)
        return actor_obs_list


    #
    #
    #
    # 构建actor观测信息
    #
    #
    #

    def get_selfloc_xy_list(self, fighter_data_obs_list):  #己方飞机位置 (x,y)
        self_location_xy_list = np.zeros((self.fighter_num, 2))
        for i in range(self.fighter_num):
            x_pos_i = fighter_data_obs_list[i].get('pos_x')
            y_pos_i = fighter_data_obs_list[i].get('pos_y')
            self_location_xy_list[i][0] = x_pos_i
            self_location_xy_list[i][1] = y_pos_i
        return self_location_xy_list


    def get_selfloc_polar_list(self, fighter_data_obs_list):  #己方飞机位置 极坐标
        self_location_xy_list = self.get_selfloc_xy_list(fighter_data_obs_list)
        self_location_polar_list = np.zeros(((self.fighter_num, self.fighter_num, 2)))

        for i in range(self.fighter_num):
            self_x_pos = self_location_xy_list[i][0]
            self_y_pos = self_location_xy_list[i][1]
            for j in range(self.fighter_num):
                if j != i:
                    if fighter_data_obs_list[i].get('alive') == True:
                        x_pos_j = self_location_xy_list[j][0]
                        y_pos_j = self_location_xy_list[j][1]
                        #距离
                        distance = pow((pow(self_x_pos - x_pos_j, 2) + pow(self_y_pos - y_pos_j, 2)), 0.5)
                        #角度
                        theta = 0
                        if self_x_pos == x_pos_j:
                            if self_y_pos > y_pos_j:
                                theta = 270
                            else:
                                theta = 90
                        else:
                            delta_x = x_pos_j - self_x_pos
                            delta_y = y_pos_j - self_y_pos
                            tan_theta = abs(delta_y) / abs(delta_x)
                            if delta_x < 0 and delta_y > 0:
                                theta = 180 - math.atan(tan_theta)
                            elif delta_x < 0 and delta_y < 0:
                                theta = 180 + math.atan(tan_theta)
                            elif delta_x > 0 and delta_y < 0:
                                theta = 360 - math.atan(tan_theta)
                            else:
                                theta = 0
                        self_location_polar_list[i][j][0] = distance
                        self_location_polar_list[i][j][1] = theta
                    else:
                        self_location_polar_list[i][j][0] = -1000
                        self_location_polar_list[i][j][1] = -360
        return self_location_polar_list


    def get_visible_list(self, fighter_data_obs_list): #主动观测到的敌机位置信息
        #主动观测
        visible_list_all = []
        for i in range(self.fighter_num):
            visible_list_i = fighter_data_obs_list[i].get('r_visible_list')
            visible_list_all.append(visible_list_i)

        visible_id_list = np.zeros((self.fighter_num, self.fighter_num))  # 敌机id
        visible_location_list = np.zeros(((self.fighter_num, self.fighter_num, 2))) #直角坐标
        for i in range(self.fighter_num):
            visible_list_i = visible_list_all[i]
            if len(visible_list_i):
                for j in range(len(visible_list_i)): #遍历飞机i观测到的敌机信息
                    enemy_id = visible_list_i[j].get('id') -1
                    enemy_x = visible_list_i[j].get('pos_x')
                    enemy_y = visible_list_i[j].get('pos_y')
                    visible_id_list[i][enemy_id] = 1
                    visible_location_list[i][enemy_id][0] = enemy_x
                    visible_location_list[i][enemy_id][1] = enemy_y
        return visible_id_list, visible_location_list


    def get_recv_list(self, fighter_data_obs_list):  #被动观测到的敌机位置信息
        #被动观测
        recv_list_all = []
        for i in range(self.fighter_num):
            recv_list_i = fighter_data_obs_list[i].get('j_recv_list')
            recv_list_all.append(recv_list_i)

        recv_id_list = np.zeros((self.fighter_num, self.fighter_num))
        recv_direction_list = np.zeros((self.fighter_num, self.fighter_num))
        for i in range(self.fighter_num):
            recv_list_i = recv_list_all[i]
            if len(recv_list_i):
                for j in range(len(recv_list_i)): #遍历飞机i观测到的敌机信息
                    enemy_id = recv_list_i[j].get('id')-1
                    enemy_direction = recv_list_i[j].get('direction')
                    recv_id_list[i][enemy_id] = 1
                    recv_direction_list[i][enemy_id] = enemy_direction
        return recv_id_list, recv_direction_list

    def get_x_id_list(self, fighter_data_obs_list): #观测到的敌机id
        x1_enemy_id_list = np.zeros((self.fighter_num, self.fighter_num))
        visible_id_list, visible_location_list = self.get_visible_list(fighter_data_obs_list)
        recv_id_list, recv_direction_list = self.get_recv_list(fighter_data_obs_list)
        for i in range(self.fighter_num):
            for j in range(self.fighter_num):
                if visible_id_list[i][j]==1 or recv_id_list[i][j]==1:
                    x1_enemy_id_list[i][j] = 1
                else:
                    x1_enemy_id_list[i][j] = 0
        return x1_enemy_id_list

    def get_missile_list(self, fighter_data_obs_list):  #飞机剩余导弹
        missile_list = np.zeros((self.fighter_num, 2))
        for i in range(self.fighter_num):
            missile_list[i][0] = fighter_data_obs_list[i].get('l_missile_left')
            missile_list[i][1] = fighter_data_obs_list[i].get('s_missile_left')

        return missile_list








    def cour_input(self, fighter_data_obs_list):
        visible_id_list, visible_location_list = self.get_visible_list(fighter_data_obs_list)
        recv_id_list, recv_direction_list = self.get_recv_list(fighter_data_obs_list)
        enemy_id_list = np.zeros(10)  #得到己方主动观测到的全部敌方id
        enemy_location_list = np.zeros((10, 2))  # 得到己方主动观测到的全部敌方位置
        for i in range(self.fighter_num):
            for j in range(self.fighter_num):
                if visible_id_list[i][j] == 1:
                    enemy_id_list[j] = 1
                    enemy_location_list[j][0] = visible_location_list[i][j][0]
                    enemy_location_list[j][1] = visible_location_list[i][j][1]

        self_location_xy_list = self.get_selfloc_xy_list(fighter_data_obs_list)
        actor_input = np.zeros(10)
        # for i in range(10):
        #     actor_input[i] = random.randint(0, 359)
        for i in range(self.fighter_num):
            dis = np.ones((10, 10))*2000
            self_xpos = self_location_xy_list[i][0]
            self_ypos = self_location_xy_list[i][1]
            for j in range(self.fighter_num):
                if enemy_id_list[j] == 1:
                    enemy_xpos = enemy_location_list[j][0]
                    enemy_ypos = enemy_location_list[j][1]
                    dis[i][j] = pow(( pow(self_xpos-enemy_xpos, 2) + pow(self_ypos-enemy_ypos, 2) ), 0.5)
            # 计算方向
            if np.min(dis[i]) != 2000 and fighter_data_obs_list[i].get('alive')==True:  # 主动观测到敌人
                attack_id = np.argmin(dis[i])
                enemy_xpos = enemy_location_list[attack_id][0]
                enemy_ypos = enemy_location_list[attack_id][1]
                theta = 0
                if self_xpos == enemy_xpos:
                    if self_ypos > enemy_ypos:
                        theta = 270
                    else:
                        theta = 90
                else:
                    delta_x = enemy_xpos - self_xpos
                    delta_y = enemy_ypos - self_ypos
                    tan_theta = abs(delta_y) / abs(delta_x)
                    if delta_x < 0 and delta_y > 0:
                        theta = 180 - math.atan(tan_theta)
                    elif delta_x < 0 and delta_y < 0:
                        theta = 180 + math.atan(tan_theta)
                    elif delta_x > 0 and delta_y < 0:
                        theta = 360 - math.atan(tan_theta)
                    else:
                        theta = 0
                actor_input[i] = theta
            elif np.max(recv_id_list[i]) == 1 and fighter_data_obs_list[i].get('alive')==True:  # 被动观测到敌人
                attack_id = np.argmax(recv_id_list[i])
                theta = recv_direction_list[i][attack_id]
                actor_input[i] = theta

        missile_list = self.get_missile_list(fighter_data_obs_list)
        for i in range(self.fighter_num):
            if missile_list[i][0] == 0 and missile_list[i][1] == 0:
                actor_input[i] += 360
        return actor_input



    def actor_input(self, fighter_data_obs_list):
        visible_id_list, visible_location_list = self.get_visible_list(fighter_data_obs_list)
        recv_id_list, recv_direction_list = self.get_recv_list(fighter_data_obs_list)
        enemy_id_list = np.zeros(10)  #得到己方主动观测到的全部敌方id
        enemy_location_list = np.zeros((10, 2))  # 得到己方主动观测到的全部敌方位置
        for i in range(self.fighter_num):
            for j in range(self.fighter_num):
                if visible_id_list[i][j] == 1:
                    enemy_id_list[j] = 1
                    enemy_location_list[j][0] = visible_location_list[i][j][0]
                    enemy_location_list[j][1] = visible_location_list[i][j][1]

        self_location_xy_list = self.get_selfloc_xy_list(fighter_data_obs_list)
        actor_input = np.zeros(10)
        # for i in range(10):
        #     actor_input[i] = random.randint(0, 359)
        for i in range(self.fighter_num):
            dis = np.ones((10, 10))*2000
            self_xpos = self_location_xy_list[i][0]
            self_ypos = self_location_xy_list[i][1]
            for j in range(self.fighter_num):
                if enemy_id_list[j] == 1:
                    enemy_xpos = enemy_location_list[j][0]
                    enemy_ypos = enemy_location_list[j][1]
                    dis[i][j] = pow(( pow(self_xpos-enemy_xpos, 2) + pow(self_ypos-enemy_ypos, 2) ), 0.5)
            # 计算方向
            if np.min(dis[i]) != 2000 and fighter_data_obs_list[i].get('alive')==True:  # 主动观测到敌人
                attack_id = np.argmin(dis[i])
                enemy_xpos = enemy_location_list[attack_id][0]
                enemy_ypos = enemy_location_list[attack_id][1]
                theta = 0
                if self_xpos == enemy_xpos:
                    if self_ypos > enemy_ypos:
                        theta = 270
                    else:
                        theta = 90
                else:
                    delta_x = enemy_xpos - self_xpos
                    delta_y = enemy_ypos - self_ypos
                    tan_theta = abs(delta_y) / abs(delta_x)
                    if delta_x < 0 and delta_y > 0:
                        theta = 180 - math.atan(tan_theta)
                    elif delta_x < 0 and delta_y < 0:
                        theta = 180 + math.atan(tan_theta)
                    elif delta_x > 0 and delta_y < 0:
                        theta = 360 - math.atan(tan_theta)
                    else:
                        theta = 0
                actor_input[i] = theta
            elif np.max(recv_id_list[i]) == 1 and fighter_data_obs_list[i].get('alive')==True:  # 被动观测到敌人
                attack_id = np.argmax(recv_id_list[i])
                theta = recv_direction_list[i][attack_id]
                actor_input[i] = theta

        missile_list = self.get_missile_list(fighter_data_obs_list)
        for i in range(self.fighter_num):
            if missile_list[i][0] == 0 and missile_list[i][1] == 0:
                actor_input[i] += 360
        return actor_input






    def critic_input(self, fighter_data_obs_list, enemy_data_obs_list):
        critic_input = np.zeros(10)
        selfloc_xy_list = self.get_selfloc_xy_list(fighter_data_obs_list) #己方全部位置信息
        enemyloc_xy_list = self.get_selfloc_xy_list(enemy_data_obs_list)  #敌方全部位置信息
        distance = np.ones((10,10))*2000
        for i in range(10):
            self_xpos = selfloc_xy_list[i][0]
            self_ypos = selfloc_xy_list[i][1]
            for j in range(10):
                if fighter_data_obs_list[i].get('alive') == True and enemy_data_obs_list[i].get('alive') == True:
                    enemy_xpos = enemyloc_xy_list[j][0]
                    enemy_ypos = enemyloc_xy_list[j][1]
                    distance[i][j] = pow((pow(self_xpos - enemy_xpos, 2) + pow(self_ypos - enemy_ypos, 2)), 0.5)
            if np.min(distance[i]) != 2000:
                nearest_enemy = np.argmin(distance[i])
                enemy_xpos = enemyloc_xy_list[nearest_enemy][0]
                enemy_ypos = enemyloc_xy_list[nearest_enemy][1]
                theta = 0
                if self_xpos == enemy_xpos:
                    if self_ypos > enemy_ypos:
                        theta = 270
                    else:
                        theta = 90
                else:
                    delta_x = enemy_xpos - self_xpos
                    delta_y = enemy_ypos - self_ypos
                    tan_theta = abs(delta_y) / abs(delta_x)
                    if delta_x < 0 and delta_y > 0:
                        theta = 180 - math.atan(tan_theta)
                    elif delta_x < 0 and delta_y < 0:
                        theta = 180 + math.atan(tan_theta)
                    elif delta_x > 0 and delta_y < 0:
                        theta = 360 - math.atan(tan_theta)
                    else:
                        theta = 0
                critic_input[i] = theta
        return critic_input













    def get_x1_list(self, fighter_data_obs_list): #观测到的敌机位置
        x2_enemy_pos_list =  np.zeros(((self.fighter_num, self.fighter_num, 2))) #极坐标
        self_location_list = self.get_selfloc_xy_list(fighter_data_obs_list)
        visible_id_list, visible_location_list = self.get_visible_list(fighter_data_obs_list)
        recv_id_list, recv_direction_list = self.get_recv_list(fighter_data_obs_list)
        for i in range(self.fighter_num):
            self_xpos = self_location_list[i][0]
            # print(self_xpos)
            self_ypos = self_location_list[i][1]
            for j in range(self.fighter_num):
                if visible_id_list[i][j] == 1:
                    enemy_xpos = visible_location_list[i][j][0]
                    enemy_ypos = visible_location_list[i][j][1]
                    distance = pow(( pow(self_xpos-enemy_xpos, 2) + pow(self_ypos-enemy_ypos, 2) ), 0.5)
                    #计算角度
                    theta = 0
                    if self_xpos == enemy_xpos:
                        if self_ypos > enemy_ypos:
                            theta = 270
                        else:
                            theta = 90
                    else:
                        delta_x = enemy_xpos - self_xpos
                        delta_y = enemy_ypos - self_ypos
                        tan_theta = abs(delta_y)/abs(delta_x)
                        if delta_x<0 and delta_y>0:
                            theta = 180 - math.atan(tan_theta)
                        elif delta_x<0 and delta_y<0:
                            theta = 180 + math.atan(tan_theta)
                        elif delta_x>0 and delta_y<0:
                            theta = 360 - math.atan(tan_theta)
                        else:
                            theta = 0

                    x2_enemy_pos_list[i][j][0] = distance
                    x2_enemy_pos_list[i][j][1] = theta
                else:
                    x2_enemy_pos_list[i][j][0] = -1000
                    x2_enemy_pos_list[i][j][1] = -360

        for i in range(self.fighter_num):
            for j in range(self.fighter_num):
                if visible_id_list[i][j] == 0 and recv_id_list[i][j] == 1:
                    theta = recv_direction_list[i][j]
                    x2_enemy_pos_list[i][j][0] = 200  #超出主动观测范围的距离
                    x2_enemy_pos_list[i][j][1] = theta
        return x2_enemy_pos_list



    def get_x2_list(self, fighter_data_obs_list):  #己方飞机位置
        # return self.get_selfloc_xy_list(fighter_data_obs_list)
        return self.get_selfloc_polar_list(fighter_data_obs_list)

    def get_x3_list(self, fighter_data_obs_list):  #探测雷达
        self_radar_list = np.zeros(self.fighter_num)
        for i in range(self.fighter_num):
            r_iswork_i = fighter_data_obs_list[i].get('r_iswork')  #开关
            r_fre_point_i = fighter_data_obs_list[i].get('r_fre_point')  #频点
            self_radar_list[i] = r_iswork_i*r_fre_point_i
        return self_radar_list

    def get_x4_list(self, fighter_data_obs_list):  #干扰雷达
        self_jam_list = np.zeros(self.fighter_num)
        for i in range(self.fighter_num):
            j_iswork_i = fighter_data_obs_list[i].get('j_iswork')  #开关
            j_fre_point_i = fighter_data_obs_list[i].get('j_fre_point')  #频点
            self_jam_list[i] = j_iswork_i*j_fre_point_i
        return self_jam_list



    def get_x5_list(self, fighter_data_obs_list):  #剩余导弹数量
        self_missile_list = np.zeros((self.fighter_num, 2))
        for i in range(self.fighter_num):
            l_missile_i = fighter_data_obs_list[i].get('l_missile_left')
            # print(type(l_missile_i))
            # print(l_missile_i)
            s_missile_i = fighter_data_obs_list[i].get('s_missile_left')
            self_missile_list[i][0] = l_missile_i
            self_missile_list[i][1] = s_missile_i
        return self_missile_list





    #
    #
    #
    # 构建actor观测信息
    #
    #
    #
    def get_total_info(self, fighter_data_obs_list):
        total_info = []
        for i in range(self.fighter_num):
            agent_info = np.zeros(20)
            if fighter_data_obs_list[i].get('alive') == True:
                agent_info[0] = 1
                agent_info[1] = fighter_data_obs_list[i].get('pos_x')
                agent_info[2] = fighter_data_obs_list[i].get('pos_y')
                agent_info[3] = fighter_data_obs_list[i].get('course')
                agent_info[4] = fighter_data_obs_list[i].get('l_missile_left')
                agent_info[5] = fighter_data_obs_list[i].get('s_missile_left')
                agent_info[6] = fighter_data_obs_list[i].get('r_iswork')
                agent_info[7] = fighter_data_obs_list[i].get('r_fre_point')
                agent_info[8] = fighter_data_obs_list[i].get('j_iswork')
                agent_info[9] = fighter_data_obs_list[i].get('j_fre_point')
                visible_list_i = fighter_data_obs_list[i].get('r_visible_list')
                recv_list_i = fighter_data_obs_list[i].get('j_recv_list')
                if len(visible_list_i):
                    for j in range(len(visible_list_i)):  # 遍历飞机i主动观测到的敌机信息
                        enemy_id = visible_list_i[j].get('id') - 1
                        agent_info[10+enemy_id] = 1
                if len(recv_list_i):
                    for j in range(len(recv_list_i)):  # 遍历飞机i被动观测到的敌机信息
                        enemy_id = recv_list_i[j].get('id') - 1
                        agent_info[10+enemy_id] = 1
            total_info.append(agent_info)
        return(total_info)

