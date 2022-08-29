import numpy as np
import random
from agent.fix_rule.agent import Agent
from interface import Environment
import math



def get_visible_list(fighter_data_obs_list):  # 主动观测到的敌机位置信息
    # 主动观测
    visible_list_all = []
    for i in range(10):
        visible_list_i = fighter_data_obs_list[i].get('r_visible_list')
        visible_list_all.append(visible_list_i)

    visible_id_list = np.zeros((10, 10))  # 敌机id
    visible_location_list = np.zeros(((10, 10, 2)))  # 直角坐标
    for i in range(10):
        visible_list_i = visible_list_all[i]
        if len(visible_list_i):
            for j in range(len(visible_list_i)):  # 遍历飞机i观测到的敌机信息
                enemy_id = visible_list_i[j].get('id') - 1
                enemy_x = visible_list_i[j].get('pos_x')
                enemy_y = visible_list_i[j].get('pos_y')
                visible_id_list[i][enemy_id] = 1
                visible_location_list[i][enemy_id][0] = enemy_x
                visible_location_list[i][enemy_id][1] = enemy_y
    return visible_id_list, visible_location_list



def actor_input(fighter_data_obs_list):
    actor_input = np.zeros(10)
    visible_id_list, visible_location_list = get_visible_list(fighter_data_obs_list)

    for i in range(10):
        distance_i = np.ones(10)*1000
        self_xpos = fighter_data_obs_list[i].get('pos_x')
        self_ypos = fighter_data_obs_list[i].get('pos_y')
        for j in range(10):
            if visible_id_list[i][j] != 0:
                distance_i[j] = pow((pow(self_xpos - visible_location_list[i][j][0], 2) + pow(self_ypos - visible_location_list[i][j][1], 2)), 0.5)
        distance_min = np.min(distance_i)
        enemy_id_min = np.argmin(distance_i)
        if distance_min<180:
            if (distance_min>80 or distance_min==80) and (distance_min<100 or distance_min==100):
                actor_input[i] = enemy_id_min
            elif (distance_min>100) and (distance_min<120 or distance_min==120):
                actor_input[i] = enemy_id_min + 10
            elif (distance_min>120) and (distance_min<140 or distance_min==140):
                actor_input[i] = enemy_id_min + 20
            elif (distance_min>140) and (distance_min<160 or distance_min==160):
                actor_input[i] = enemy_id_min + 30
            else:
                actor_input[i] = enemy_id_min + 40

        if actor_input[i] == 0:
            actor_input[i] = -1

    return actor_input


blue_agent = Agent()
red_agent = Agent()

blue_agent_obs_ind = blue_agent.get_obs_ind()
red_agent_obs_ind = red_agent.get_obs_ind()
print("red_agent_obs_ind:",red_agent_obs_ind)
env = Environment(
    'maps/1000_1000_fighter10v10.map',
    red_agent_obs_ind,
    blue_agent_obs_ind,
    render=False
)


for num_train in range(10):
    print("num_train:",num_train)
    size_x, size_y = env.get_map_size()
    red_detector_num, red_fighter_num, blue_detector_num, blue_fighter_num = env.get_unit_num()
    print(f"map_size x: {size_x}, map_size y: {size_y}")
    print(f"agent num: (red)-[{red_detector_num}, {red_fighter_num}] | (blue)-[{blue_detector_num}, {blue_fighter_num}]")
    red_agent.set_map_info(size_x, size_y, red_detector_num, red_fighter_num)
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
        Q_table_i = np.zeros((50, 2))
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
        Q_table_list_now = []
        for i in range(10):
            Q_table_i = np.zeros((50, 2))
            Q_table_list_now.append(Q_table_i)

        epsilon = 1-math.exp(-num_game/100)


        score = 0
        episode_step = 0
        env.reset()
        red_obs_dict, blue_obs_dict = env.get_obs()  # 当前时刻环境信息
        red_fighter_obs = red_obs_dict.get('fighter_obs_list')
        red_obs = actor_input(red_fighter_obs)  # 状态信息

        while not env.get_done():
            if episode_step < 120:
                red_detector_action, red_fighter_action = red_agent.get_action(red_obs_dict, episode_step)
                # print(red_fighter_action)
                blue_detector_action, blue_fighter_action = blue_agent.get_action(blue_obs_dict, episode_step)
                env.step(red_detector_action, red_fighter_action, blue_detector_action, blue_fighter_action)
                red_obs_dict, blue_obs_dict = env.get_obs()
            else:
                red_obs_dict, blue_obs_dict = env.get_obs()  # 当前时刻环境信息
                red_fighter_obs = red_obs_dict.get('fighter_obs_list')
                red_obs = actor_input(red_fighter_obs)
                attack_action = np.zeros(10)
                s_now_list = np.zeros(10)
                actions_now_list = np.zeros(10)

                if random.uniform(0, 1) > epsilon:  # 探索
                    explore = True
                    for i in range(10):
                        attack_action[i] = random.randint(0, 20)

                else:                               # 最优动作
                    for i in range(10):
                        s_i = int(red_obs[i])
                        if s_i > -1:
                            att_nif = np.argmax(Q_table_list[i][s_i])
                            if att_nif:
                                attack_action[i] = 0
                            elif att_nif == 0:
                                att_id = s_i%10 + 1
                                # print(att_id)
                                if red_fighter_obs[i]['l_missile_left'] > 0:
                                    attack_action[i] = att_id
                                else:
                                    attack_action[i] = att_id + 10
                        learn_id = attack_action

                for i in range(10):
                    s_now_list[i] = int(red_obs[i])
                    if attack_action[i] == 0:
                        actions_now_list[i] = 1
                    else:
                        actions_now_list[i] = 0

                red_detector_action, red_fighter_action = red_agent.get_action(red_obs_dict, episode_step)
                blue_detector_action, blue_fighter_action = blue_agent.get_action(blue_obs_dict, episode_step)
                for i in range(10):
                    if attack_action[i] == 0:
                        red_fighter_action[i]['hit_target'] = 0
                        red_fighter_action[i]['missile_type'] = 0
                    elif (attack_action[i] > 0) and (attack_action[i] < 10 or attack_action[i] == 10):
                        red_fighter_action[i]['hit_target'] = int(attack_action[i])
                        red_fighter_action[i]['missile_type'] = 1
                    elif attack_action[i] > 10:
                        red_fighter_action[i]['hit_target'] = int(attack_action[i] - 10)
                        red_fighter_action[i]['missile_type'] = 2
                    red_fighter_action[i]['r_fre_point'] = 2
                    red_fighter_action[i]['j_fre_point'] = 1

                env.step(red_detector_action, red_fighter_action, blue_detector_action, blue_fighter_action)

                red_obs_dict_, blue_obs_dict_ = env.get_obs()
                red_detector_reward, red_fighter_reward, red_game_reward, blue_detector_reward, blue_fighter_reward, blue_game_reward = env.get_reward()  # 获取奖励值
                reward_all = np.sum(red_fighter_reward + red_game_reward)
                fighter_reward = red_fighter_reward + 0.01*red_game_reward

                reward_list = fighter_reward
                Q_next_list = np.zeros(10)
                for i in range(10):
                    s_i_ = int(red_obs[i])
                    Q_next_list[i] = np.max(Q_table_list[i][s_i_])
                    s_i = int(s_now_list[i])
                    a_i = int(actions_now_list[i])
                    Q_table_list_now[i][s_i][a_i] = Q_table_list_now[i][s_i][a_i] + alpha * (reward_list[i] + gamma * Q_next_list[i] -  Q_table_list_now[i][s_i][a_i])

                red_obs_dict = red_obs_dict_
                blue_obs_dict = blue_obs_dict_

                score += reward_all

            if env.get_done() == True:
                red_fighter_dict = red_obs_dict.get('fighter_obs_list')
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
                if blue_dead == 0:
                    kill_missile = 0
                else:
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

        for i in range(10):
            for j in range(50):
                for k in range(2):
                    Q_table_list[i][j][k] = Q_table_list[i][j][k]*0.9 + Q_table_list_now[i][j][k]*0.1

        result_num = num_train+1
        result_name = str(result_num)
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

