import torch
import numpy as np
import math
from datetime import datetime
import os

import Utility
from TCPServer import Server
from Environment import Environment
# from CustomThread import CustomThread
from AgentDDPG import Agent
from UnityAdaptor import UnityAdaptor
# from Entity import State
import Entity

DEG2RAD = 0.01745329251

class Env(Environment):
    def __init__(self, max_times_in_episode, max_times_in_game, end_distance, stop_target, target_fixed_sec):
        super().__init__(max_times_in_episode, max_times_in_game, end_distance, stop_target, target_fixed_sec)
    
    # check episode termination
    def check_termination(self):
        distance = math.dist(self.pos, self.target_pos)
        self.reach_goal = ((abs(self.carOrientation - self.trailOrientation) < 5) \
            and distance < self.end_distance[0])

        self.distance_out = distance > self.end_distance[1] or distance < self.end_distance[0]
        self.game_finished = self.game_ctr > self.max_times_in_game

        if self.reach_goal:
            print("reach_goal!!!!!!!!!!!!!!!!!!!")
        # if self.episode_ctr >= self.max_times_in_episode:
        #     print("episode ctr >= {}".format(self.max_times_in_episode))
        if self.game_finished:
            print("game_ctr >= {}".format(self.max_times_in_game))
        if distance <= self.end_distance[0]:
            print("distance <= {}".format(self.end_distance[0]))
        if distance >= self.end_distance[1]:
            print("distance >= {}".format(self.end_distance[1]))


        done = self.reach_goal or self.distance_out \
            or self.episode_ctr > self.max_times_in_episode \
            or self.game_finished
        return done, self.reach_goal

    def calculate_reward(self, state: Entity.State, new_state: Entity.State):
        reward = 0

        self.pos = [new_state.car_pos.x, new_state.car_pos.y]
        self.prev_pos = [state.car_pos.x, state.car_pos.y]

        self.carOrientation = Utility.rad2deg(new_state.car_orientation)
        prevCarOrientation = Utility.rad2deg(state.car_orientation)

        self.trailOrientation = Utility.rad2deg(Utility.radFromUp([new_state.path_closest_pos.x, \
            new_state.path_closest_pos.y], [new_state.path_second_pos.x, \
            new_state.path_second_pos.y]))
        
        ###############TODO###############
        ### distance to far trail
        prevTrailPos = [state.path_farthest_pos.x, state.path_farthest_pos.y]
        prevTrailDist = self.calculate_distance(self.prev_pos, prevTrailPos)
        distanceToTrail = self.calculate_distance(self.pos, prevTrailPos)
        distanceDiff = distanceToTrail - prevTrailDist
        distanceDiff *= 800
        reward += -distanceDiff
        # print("target d", -distanceDiff)
     
        # print("total ", reward)
        # print("----------------------")

        return reward
    
    def step(self, state, new_state):
        self.episode_ctr += 1
        self.game_ctr += 1
        self.total_ctr += 1
        
        reward = self.calculate_reward(state, new_state)
        
        done, reachGoal = self.check_termination() #self.trailOrientation

        if reachGoal:
            reward += 400
        
        info = {'prev pos': []}
        info['prev pos'] = self.prev_pos
        info['trail original pos'] = [0, 0]

        return reward, done, info

class Agt(Agent):
    def __init__(self, q_lr, pi_lr, gamma, rho, 
                 pretrained, new_input_dims, input_dims, n_actions,
                 layer1_size, layer2_size, batch_size, chpt_dir_load, chpt_dir_save):
        super().__init__(q_lr, pi_lr, gamma, rho, 
                    pretrained, new_input_dims, input_dims, n_actions,
                 layer1_size, layer2_size, batch_size, chpt_dir_load, chpt_dir_save)

    def choose_actions(self, obs, prev_pos, trail_original_pos, inference):
        self.actor.eval()
        obs = self.processFeature(obs, prev_pos, trail_original_pos)
        obs = torch.tensor(obs, dtype = torch.float).to(self.device)
        with torch.no_grad():
            actions = self.actor.forward(obs).to(self.device)
        self.actor.train()

        # noice
        if not inference:
            actions = actions + torch.tensor(self.noice(), dtype = torch.float).to(self.device)
    
        return actions.cpu().detach().numpy()
    
    
    def processFeature(self, state: Entity.State, prev_pos, trail_original_pos):
        feature = []
   
        # distance between car and target
        # feature.append(state['car pos']['x']- state['final target pos']['x']) 
        # feature.append(state.car_pos.y- state['final target pos']['y'])
        
        # distance between car and trail
        feature.append(state.car_pos.x - state.path_closest_pos.x) #old: state.car_pos.y, -state.car_pos.x
        feature.append(state.car_pos.y - state.path_closest_pos.y) #old: state.path_closest_pos.y, -state.path_closest_pos.x
        # print("vector")
        # print(state.car_pos.x - state.path_closest_pos.x, state.car_pos.y - state.path_closest_pos.y)
        # print(state.car_pos.y - state.path_closest_pos.y, -state.car_pos.x+state.path_closest_pos.x)

        # distance between car and far trail(target)
        feature.append(state.car_pos.x- state.path_farthest_pos.x) 
        feature.append(state.car_pos.y- state.path_farthest_pos.y)

        # angle in radian between up(0, 1)vector and car to far trail(target)
        rad = Utility.radFromUp([state.car_pos.x, state.car_pos.y], \
            [state.path_farthest_pos.x, state.path_farthest_pos.y])
        feature.append(Utility.decomposeCosSin(rad)) #cos(radian), sin(radian) *2
        # print("car to target: ", rad / DEG2RAD)
        
        # angle in radian between up and trail slope
        rad = Utility.radFromUp([state.path_closest_pos.x, state.path_closest_pos.y], \
        [state.path_second_pos.x, state.path_second_pos.y])
        feature.append(Utility.decomposeCosSin(rad)) 
        
        # car orientation(eular angles in radians)
        feature.append(Utility.decomposeCosSin(state.car_orientation)) 
        # print("car: ", state.car_orientation / DEG2RAD)
        
        # car velocity
        feature.append(state.car_vel.x)  
        feature.append(state.car_vel.y)

        # car angular velocity in radians(eular angles in radians)
        feature.append(state.car_angular_vel)

        # 前進軸 angular velocity in radians *4 --> *1
        feature.append(state.wheel_angular_vel.left_back)
        # feature.append(state.wheel_angular_vel.left_front)
        # feature.append(state.wheel_angular_vel.right_back)
        # feature.append(state.wheel_angular_vel.right_front)


        # 轉動軸 in radians
        # feature.append(utility.decomposeCosSin(state.wheel_orientation.left_back)) 
        feature.append(Utility.decomposeCosSin(state.wheel_orientation.left_front))
        # feature.append(utility.decomposeCosSin(state.wheel_orientation.right_back))
        # feature.append(utility.decomposeCosSin(state.wheel_orientation.right_front))
        

        # min lidar in meters
        # feature.append(state.min_lidar)

        feature = Utility.flatten(feature)
        return feature

def main(mode):
    print('The mode is:', mode)

    # TODO paramaterization
    server = Server(port=5070) #global angular v: 5070, local: 5090
    # t = CustomThread(server)
    env = Env(max_times_in_episode=50, max_times_in_game=250, end_distance=(0.2, 6), stop_target=False, target_fixed_sec=12)
  
    chpt_dir_load = os.path.join(os.path.dirname(__file__), './Model', '0507_curve_rear_engine_car_new_protocal_local_angularV')
    chpt_dir_save = os.path.join(os.path.dirname(__file__), './Model', '0516_curve_rear_engine_car_new_protocal/model')
    chpt_dir_plot = os.path.join(os.path.dirname(__file__), './Model', '0516_curve_rear_engine_car_new_protocal/')

    agent = Agt(q_lr=0.001, pi_lr=0.001, gamma=0.99, rho=0.005,  \
        pretrained=False, new_input_dims=17, \
        input_dims=16, n_actions=2, batch_size=100, layer1_size=400, layer2_size=300, \
        chpt_dir_load=chpt_dir_load, chpt_dir_save=chpt_dir_save)
    
    epoch = 10000

    reward_history, reward_history_ = ([] for i in range(2))

    unity_adaptor = UnityAdaptor(action_range=1200, steering_angle_range=20)

    state = Entity.State(final_target_pos=Entity.ROS2Point(x=0.0, y=0.0, z=0.0),
                car_pos=Entity.ROS2Point(x=0.0, y=0.0, z=0.0),
                path_closest_pos=Entity.ROS2Point(x=0.0, y=0.0, z=0.0),
                path_second_pos=Entity.ROS2Point(x=0.0, y=0.0, z=0.0),
                path_farthest_pos=Entity.ROS2Point(x=0.0, y=0.0, z=0.0),
                car_vel=Entity.ROS2Point(x=0.0, y=0.0, z=0.0),
                car_orientation=0.0,
                wheel_orientation=Entity.WheelOrientation(left_front=0.0, right_front=0.0),
                car_angular_vel=0.0,
                wheel_angular_vel=Entity.WheelAngularVel(left_back=0.0, left_front=0.0, right_back=0.0, right_front=0.0),
                min_lidar=0.0,
                min_lidar_direciton=Entity.ROS2Point(x=0.0, y=0.0, z=0.0),
                min_lidar_relative_angle=0.0,
                action_wheel_angular_vel = Entity.WheelAngularVel(left_back=0.0, left_front=0.0, right_back=0.0, right_front=0.0),
                action_wheel_orientation = Entity.WheelOrientation(left_front=0.0, right_front=0.0)
                )

    try:
        if mode == 'train':
            lr_c_history, lr_a_history, critic_loss_history, actor_loss_history = (
                [] for i in range(4))
            
            unity_obs = None
            
            # TODO paramaterization
            load_step = 800  # 0
            agent.load_models(load_step)

            # elapsed_time = 0.2  # s 0.06
            # prev_time_step = datetime.now().microsecond/1000000 - elapsed_time

            prev_pos = [0, 0]
            trail_original_pos = [0, 0]

            ai_action = [0, 0]

            for i in range(load_step+1, load_step+epoch+1):
                if unity_obs == None:
                    while unity_obs is None:
                        unity_obs = server.recvData()
                    state = unity_adaptor.transfer_obs(unity_obs, ai_action)    
                    env.restart_game(state)
                else:
                    restart_game = env.restart_episode()
                    if restart_game:
                        new_target = {'title': 'new target', 'content': {}}
                        server.sendAction(new_target)
                        unity_obs = server.recvData()
                        state = unity_adaptor.transfer_obs(unity_obs, ai_action)    
                        env.restart_game(state)

                done = False
                score, lr_c, lr_a = (0 for i in range(3))
                c_loss, a_loss = ([] for i in range(2))

                while (not done):
                    ai_action = agent.choose_actions(state, prev_pos, trail_original_pos, inference=False)

                    action_sent_to_unity = unity_adaptor.trasfer_action(ai_action)
                    server.sendAction(action_sent_to_unity)

                    # time.sleep(0.2) ######

                    # calculate time
                    # timeStep = datetime.now().microsecond/1000000 #seconds
                    # elapsed_time = timeStep - prev_time_step
                    # if elapsed_time < 0:
                    #     elapsed_time += 1
                    # # print(elapsed_time)
                    # prev_time_step = datetime.now().microsecond/1000000

                    unity_new_obs = server.recvData()
                    new_state = unity_adaptor.transfer_obs(unity_new_obs, ai_action)
                    
                    reward, done, info = env.step(state, new_state)
                    
                    score += reward

                    agent.store_transition(state, ai_action, reward, new_state, int(done), prev_pos, trail_original_pos)

                    prev_pos = info['prev pos']
                    trail_original_pos = info['trail original pos']

                    state = new_state

                for j in range(env.episode_ctr):
                    c_loss_, a_loss_, lr_c, lr_a = agent.learn()
                    c_loss.append(c_loss_)
                    a_loss.append(a_loss_)

                reward_history.append(score)
                lr_c_history.append(lr_c)
                lr_a_history.append(lr_a)
                critic_loss_history.append(Utility.mean(c_loss))
                actor_loss_history.append(Utility.mean(a_loss))
                reward_history_.append(np.mean(reward_history[-50:]))

                if (i) % 200 == 0:
                    agent.save_models(i)
                    Utility.plot(reward_history_, lr_c_history, lr_a_history, critic_loss_history,
                                actor_loss_history, i, path=chpt_dir_plot)

                print('episode:', i,
                    ',reward:%.2f' % (score),
                    ',avg reward:%.2f' % (reward_history_[-1]),
                    ',critic loss:%.2f' % (critic_loss_history[-1]),
                    ',actor loss:%.2f' % (actor_loss_history[-1]),
                    ',ctr:', env.episode_ctr,)

        elif mode == 'test':
            agent.load_models(800)
            agent.eval()
            unity_obs = None
            prev_pos = [0, 0]
            trail_original_pos = [0, 0]

            ai_action = [0, 0]

            for i in range(1000):

                restart_game = env.restart_episode()
                if unity_obs == None:
                    while unity_obs is None:
                        unity_obs = server.recvData()
                    state = unity_adaptor.transfer_obs(unity_obs, ai_action)    
                    env.restart_game(state)
                else:
                    restart_game = env.restart_episode()
                    if restart_game:
                        new_target = {'title': 'new target', 'content': {}}
                        server.sendAction(new_target)
                        unity_obs = server.recvData()
                        state = unity_adaptor.transfer_obs(unity_obs, ai_action)    
                        env.restart_game(state)

                done = False
                score = 0

                while (not done):
                    ai_action = agent.choose_actions(state, prev_pos, trail_original_pos, inference=True)

                    action_sent_to_unity = unity_adaptor.trasfer_action(ai_action)
                    server.sendAction(action_sent_to_unity)

                    unity_new_obs = server.recvData()
                    new_state = unity_adaptor.transfer_obs(unity_new_obs, ai_action)

                    reward, done, info = env.step(state, new_state)

                    score += reward
                    prev_pos = info['prev pos']
                    trail_original_pos = info['trail original pos']

                    state = new_state

                print('test run: ', i, ',reward: %.2f,' % (score), 'ctr: ', env.game_ctr)

        server.close()

    except KeyboardInterrupt:
        server.close()

if __name__ == '__main__':
    #TODO paramaterization
    mode = 'train'
    # mode = 'test'
    main(mode)
