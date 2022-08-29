import numpy as np
import random
from agent.fix_rule.agent import Agent
from interface import Environment
import math

import os
print(os.path.abspath("."))


blue_agent = Agent()

blue_agent_obs_ind = blue_agent.get_obs_ind()
red_agent_obs_ind = 'maddpg'
env = Environment(
    'maps/1000_1000_fighter10v10.map',
    red_agent_obs_ind,
    blue_agent_obs_ind,
    render=False
)


for num_train in range(50):

    print("num_train:",num_train)
    size_x, size_y = env.get_map_size()
    red_detector_num, red_fighter_num, blue_detector_num, blue_fighter_num = env.get_unit_num()
    print(f"map_size x: {size_x}, map_size y: {size_y}")
    print(f"agent num: (red)-[{red_detector_num}, {red_fighter_num}] | (blue)-[{blue_detector_num}, {blue_fighter_num}]")
    blue_agent.set_map_info(size_x, size_y, blue_detector_num, blue_fighter_num)




    PRINT_INTERVAL = 1
    N_GAMES = 1500
    total_steps = 0
    score_history = []
    red_alive_history = []
    red_dead_history = []
    blue_alive_history = []
    blue_dead_history = []
    kill_missile_history = []
    winning_rate_history = []
    best_score = 0
    alpha = 0.01
    gamma = 0.9

    Q_table_list = []
    for i in range(10):
        Q_table_i = np.zeros((720, 3))
        Q_table_list.append(Q_table_i)

    print("Q_table establish")

    SCORES = np.zeros(150)
    RED_ALIVE = np.zeros(150)
    RED_DEAD = np.zeros(150)
    BLUE_ALIVE = np.zeros(150)
    BLUE_DEAD = np.zeros(150)
    KILL_MISSILE = np.zeros(150)
    WINNING_RATE = np.zeros(150)

    for num_game in range(N_GAMES):
        epsilon = 1-math.exp(-num_game/100)
        score = 0
        episode_step = 0
        env.reset()
        red_raw_dict, red_obs, blue_obs_dict = env.get_obs()  # 当前时刻环境信息

        print("num_game:",num_game)

        while not env.get_done():
            side1_obs_fighter = red_raw_dict.get('fighter_obs_list')
            if episode_step < 129:
                red_fighter_action = np.zeros((10, 4))
                for i in range(10):
                    red_fighter_action[i][0] = 0
                    red_fighter_action[i][1] = 2
                    red_fighter_action[i][2] = 1
                    red_fighter_action[i][3] = 0
                red_detector_action = []
                blue_detector_action, blue_fighter_action = blue_agent.get_action(blue_obs_dict, episode_step)
                env.step(red_detector_action, red_fighter_action, blue_detector_action, blue_fighter_action)
                red_raw_dict, red_obs, blue_obs_dict = env.get_obs()
            else:
                blue_detector_action, blue_fighter_action = blue_agent.get_action(blue_obs_dict, episode_step)
                side1_fighter_action = np.zeros((10, 4))
                for i in range(10):
                    side1_fighter_action[i][1] = 2
                    side1_fighter_action[i][2] = 1
                    self_xpos = side1_obs_fighter[i].get('pos_x')
                    self_ypos = side1_obs_fighter[i].get('pos_y')
                    r_visible_list = side1_obs_fighter[i]['r_visible_list']
                    j_recv_list = side1_obs_fighter[i]['j_recv_list']
                    if r_visible_list != []:
                        l_missile_left = side1_obs_fighter[i]['l_missile_left']
                        enemy_distance = np.zeros(len(r_visible_list))
                        for num_enemy in range(len(r_visible_list)):
                            enemy_xpos = r_visible_list[num_enemy].get('pos_x')
                            enemy_ypos = r_visible_list[num_enemy].get('pos_y')
                            enemy_distance[num_enemy] = pow((pow(self_xpos - enemy_xpos, 2) + pow(self_ypos - enemy_ypos, 2)), 0.5)
                        attack_enemy = np.argmin(enemy_distance)
                        print("min_distance:",np.min(enemy_distance))
                        attack_id = r_visible_list[attack_enemy].get('id')
                        if side1_obs_fighter[i].get('l_missile_left') > 0:
                            side1_fighter_action[i][3] = attack_id
                        else:
                            side1_fighter_action[i][3] = attack_id + 10
                    elif j_recv_list != []:
                        p_attack = random.uniform(0, 1)
                        if p_attack > 0.8:
                            attack_id = j_recv_list[0].get('id')
                            if side1_obs_fighter[i].get('l_missile_left') > 0:
                                side1_fighter_action[i][3] = attack_id

                actions_now_list = np.zeros(10)
                Q_now_list = np.zeros(10)
                s_now_list = np.zeros(10)

                Explore = False
                if random.uniform(0, 1) > epsilon:
                    Explore = True
                for i in range(10):
                    s_i = int(red_obs[i])
                    s_now_list[i] = s_i
                    if Explore == False:
                        action_i = int(np.argmax(Q_table_list[i][s_i]))
                    else:
                        action_i = random.randint(0, 2)
                    actions_now_list[i] = action_i
                    Q_now_list[i] = Q_table_list[i][s_i][action_i]

                    # action_i = 0
                    if s_i > 359:
                        s_i -= 360
                    if action_i == 0:    #追击
                        side1_fighter_action[i][0] = s_i
                    elif action_i == 1:  #躲避
                        if s_i < 180:
                            side1_fighter_action[i][0] = s_i + 180
                        else:
                            side1_fighter_action[i][0] = s_i -180
                    elif action_i == 2:  #寻找
                        side1_fighter_action[i][0] = random.randint(0, 359)

                for i in range(10):
                    side1_obs_fighter[i].get('pos_x')

                red_fighter_action = side1_fighter_action
                red_detector_action = []
                env.step(red_detector_action, red_fighter_action, blue_detector_action, blue_fighter_action)
                red_raw_dict_, red_obs_, blue_obs_dict_ = env.get_obs()
                red_detector_reward, red_fighter_reward, red_game_reward, blue_detector_reward, blue_fighter_reward, blue_game_reward = env.get_reward()  # 获取奖励值
                fighter_reward = red_fighter_reward + red_game_reward
                reward_list = np.zeros(10)
                total_reward_w = num_game/N_GAMES
                self_reward_w = 1 - total_reward_w
                for i in range(10):
                    reward_list[i] = 10 * red_fighter_reward[i] + (np.sum(red_fighter_reward) - red_fighter_reward[i] + np.sum(red_game_reward))
                    # reward_list[i] = red_fighter_reward[i]
                # fighter_reward = np.mean(red_fighter_reward + red_game_reward)
                reward_all = np.sum(red_fighter_reward + red_game_reward)
                reward_list = fighter_reward
                Q_next_list = np.zeros(10)
                for i in range(10):
                    s_i_ = int(red_obs_[i])
                    Q_next_list[i] = np.max(Q_table_list[i][s_i_])
                    s_i = int(s_now_list[i])
                    a_i = int(actions_now_list[i])
                    Q_table_list[i][s_i][a_i] = Q_table_list[i][s_i][a_i] + alpha * (reward_list[i] + gamma * Q_next_list[i] - Q_table_list[i][s_i][a_i])

                best_action = np.zeros((10,720))  # 每个用户不同状态选择的最优动作
                for i in range(10):
                    for j in range(720):
                        best_action[i][j] = np.argmax(Q_table_list[i][j])
                action_num = np.zeros((720,3))  # 不同动作被选择的数量
                for j in range(720):
                    for i in range(10):
                        action_num[j][int(best_action[i][j])] += 1
                for i in range(10):
                    for j in range(720):
                        if best_action[i][j] == np.argmin(action_num[j]):
                            Q_table_list[i][j] = np.zeros(3)

                red_raw_dict = red_raw_dict_
                red_obs = red_obs_
                blue_obs_dict = blue_obs_dict_

                score += reward_all

            if env.get_done() == True:
                red_fighter_dict = red_raw_dict.get('fighter_obs_list')
                blue_fighter_dict = blue_obs_dict.get('fighter_obs_list')
                red_alive = 0
                red_dead = 0
                blue_alive = 0
                blue_dead = 0
                kill_missile = 0
                missile_use = 0
                win_if = 0
                if np.sum(red_fighter_reward+red_game_reward) > np.sum(blue_fighter_reward+blue_game_reward):
                    win_if = 1
                for num in range(10):
                    if red_fighter_dict[num].get('alive') == True:
                        red_alive += 1
                    else:
                        red_dead += 1
                    if blue_fighter_dict[num].get('alive') == True:
                        blue_alive += 1
                    else:
                        blue_dead += 1
                    missile_left = red_fighter_dict[num].get('l_missile_left') + red_fighter_dict[num].get('s_missile_left')
                    missile_use += (6 - missile_left)
                kill_missile = blue_dead/missile_use

            total_steps += 1
            episode_step += 1

        red_alive_history.append(red_alive)
        red_dead_history.append(red_dead)
        blue_alive_history.append(blue_alive)
        blue_dead_history.append(blue_dead)
        kill_missile_history.append(kill_missile)
        winning_rate_history.append((win_if*100))
        score_history.append(score)

        if (num_game+1)%10 == 0:
            SCORES[int((num_game+1)/10)-1] = np.mean(score_history[-10:])
            RED_ALIVE[int((num_game + 1) / 10) - 1] = np.mean(red_alive_history[-10:])
            RED_DEAD[int((num_game + 1) / 10) - 1] = np.mean(red_dead_history[-10:])
            BLUE_ALIVE[int((num_game + 1) / 10) - 1] = np.mean(blue_alive_history[-10:])
            BLUE_DEAD[int((num_game + 1) / 10) - 1] = np.mean(blue_dead_history[-10:])
            KILL_MISSILE[int((num_game + 1) / 10) - 1] = np.mean(kill_missile_history[-10:])
            WINNING_RATE[int((num_game + 1) / 10) - 1] = np.mean(winning_rate_history[-10:])
            print("num_game:",num_game+1)
            print("SCORES:", SCORES[int((num_game + 1) / 10) - 1])
            print("red_alive:",RED_ALIVE[int((num_game+1)/10)-1])
            print("red_dead:", RED_DEAD[int((num_game + 1) / 10) - 1])
            print("blue_alive:", BLUE_ALIVE[int((num_game + 1) / 10) - 1])
            print("blue_dead:", BLUE_DEAD[int((num_game + 1) / 10) - 1])
            print("kill_missile:", KILL_MISSILE[int((num_game + 1) / 10) - 1])
            print("winning_rate:", WINNING_RATE[int((num_game + 1) / 10) - 1])


    # result_num = num_train+1
    # result_name = str(result_num)
        result_name = '_best'
        if SCORES[int((num_game+1)/10)-1] > 4000:
            address1 = './score'
            address2 = './Q_table_list'
            address3 = './red_alive'
            address4 = './red_dead'
            address5 = './blue_alive'
            address6 = './blue_dead'
            address7 = './kill_missile'
            address8 = './winning_rate'

            name1 = address1 + result_name
            name2 = address2 + result_name
            name3 = address3 + result_name
            name4 = address4 + result_name
            name5 = address5 + result_name
            name6 = address6 + result_name
            name7 = address7 + result_name
            name8 = address8 + result_name

            # np.save(name1, SCORES)
            # np.save(name2, Q_table_list)
            # np.save(name3, RED_ALIVE)
            # np.save(name4, RED_DEAD)
            # np.save(name5, BLUE_ALIVE)
            # np.save(name6, BLUE_DEAD)
            # np.save(name7, KILL_MISSILE)
            # np.save(name8, WINNING_RATE)




