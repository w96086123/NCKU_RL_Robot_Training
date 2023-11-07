import json
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
import random

import time

import threading
import sys
from rclpy.node import Node
import rclpy
from std_msgs.msg import Float32MultiArray
from std_msgs.msg import String

DEG2RAD = 0.01745329251

unityState = list()

# Seed
seed = 123
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
np.random.seed(seed)
random.seed(seed)
torch.backends.cudnn.benchmark = False
torch.backends.cudnn.deterministic = True

class AiNode(Node):
    def __init__(self):
        super().__init__("aiNode")
        self.get_logger().info("Ai start")#ros2Ai #unity2Ros
        self.subsvriber_ = self.create_subscription(String, "unity2Ros", self.receive_data_from_ros, 10)
        
        self.publisher_Ai2ros = self.create_publisher(Float32MultiArray, 'ros2Unity', 10)#Ai2ros #ros2Unity
        

    
    def publish2Ros(self, data):
        self.data2Ros = Float32MultiArray()
        self.data2Ros.data = data
        self.publisher_Ai2ros.publish(self.data2Ros)

    def receive_data_from_ros(self, msg):
        global unityState        
        unityState = msg.data
        # print(unityState)
        # self.unityState = msg.data


def spin_props(node):
    exe = rclpy.executors.SingleThreadedExecutor()
    exe.add_node(node)
    exe.spin()
    rclpy.shutdown()
    sys.exit(0)

def returnUnityState():
    while len(unityState) == 0:
        pass
    return unityState

class Env(Environment):
    def __init__(self, max_times_in_episode, max_times_in_game, end_distance, stop_target, target_fixed_sec):
        super().__init__(max_times_in_episode, max_times_in_game, end_distance, stop_target, target_fixed_sec)
        self.stucked_count = 0
    # check episode termination
    def check_termination(self, state):
        
        print(state.min_lidar_direciton)
        print(state.min_lidar)

        self.pos = [state.car_pos.x, state.car_pos.y]
        self.target_pos = [state.final_target_pos.x, state.final_target_pos.y]

        distance = math.dist(self.pos, self.target_pos)
        # self.reach_goal = ((abs(self.carOrientation - self.targetOrientation) <= 20) \
        #     and distance <= self.end_distance[0])
        # self.reach_goal = (abs(self.carOrientation - self.targetOrientation) < 5) 
        self.reach_goal = (distance <= self.end_distance[0])
        self.distance_out = distance >= self.end_distance[1] or distance <= self.end_distance[0]
        self.game_finished = self.game_ctr >= self.max_times_in_game

        if self.reach_goal:
            print("reach_goal!!!!!!!!!!!!!!!!!!!")
        # if self.episode_ctr >= self.max_times_in_episode:
        #     print("episode ctr >= {}".format(self.max_times_in_episode))
        if self.game_finished:
            print("game_ctr >= {}".format(self.max_times_in_game))
        if distance >= self.end_distance[1]:
            print("distance >= {}".format(self.end_distance[1]))


        done = self.reach_goal or self.distance_out \
            or self.episode_ctr >= self.max_times_in_episode \
            or self.game_finished
        return done, self.reach_goal

    def calculateAngle(self,A, B, C):
        # 计算向量AB和BC
        vectorAB = [int((B[0] - A[0])*1000), int((B[1] - A[1])*1000)]
        vectorBC = [int((C[0] - B[0])*1000), int((C[1] - B[1])*1000)]
        print("A",vectorAB)
        print("B",vectorBC)
        # 如果向量AB和BC的长度为0，夹角为0度
        if vectorAB == [0, 0] and vectorBC == [0, 0]:
            return 0

        # 计算AB和BC的长度
        lengthAB = math.sqrt(vectorAB[0] ** 2 + vectorAB[1] ** 2)
        lengthBC = math.sqrt(vectorBC[0] ** 2 + vectorBC[1] ** 2)

        # 如果其中一个向量长度为0，夹角为180度
        if lengthAB == 0 or lengthBC == 0:
            return 180

        # 否则，计算夹角
        dot_product = vectorAB[0] * vectorBC[0] + vectorAB[1] * vectorBC[1]
        cos_theta = dot_product / (lengthAB * lengthBC)

        # 修正cosine值范围
        if cos_theta < -1:
            cos_theta = -1
        elif cos_theta > 1:
            cos_theta = 1

        # 计算角度
        angle_rad = math.acos(cos_theta)
        degrees = int(angle_rad * (180 / math.pi))
        return degrees


    def calculate_reward(self, state: Entity.State, new_state: Entity.State):
        reward = 0

        self.pos = [new_state.car_pos.x, new_state.car_pos.y]
        self.prev_pos = [state.car_pos.x, state.car_pos.y]
        # 計算目前車子方向，以我們自己建的180為前進0為後退
        self.carOrientation = Utility.rad2deg(new_state.car_orientation)
        prevCarOrientation = Utility.rad2deg(state.car_orientation)
        # print(state.car_orientation)
        target_pos = [state.final_target_pos.x , state.final_target_pos.y]

        self.targetOrientation = Utility.rad2deg(Utility.radFromUp(self.pos, target_pos))
    
        ### distance to final target
        # 計算方式為 (distance[T-1])-(distance[T])
        # 若距離減短為加分，但為正數時為不行
        prevTargetDist = self.calculate_distance(self.prev_pos, target_pos)
        distanceToTarget = self.calculate_distance(self.pos, target_pos)

        distanceDiff = distanceToTarget - prevTargetDist
        print("distance : ",distanceDiff)
        if distanceDiff > 0:
            distanceDiff *= 2
        distanceDiff *= 800
        reward += -distanceDiff

        # 直接計算車子與目標的角度方向，若為0則代表方向一次，180則為反方向
        angle = self.calculateAngle(self.prev_pos,self.pos,target_pos)
        targetAngleDiff = angle
        print("angle", angle)
        
        targetAngleDiff *= 1
        reward += -targetAngleDiff

    

        ###
        if ((self.game_ctr - 1) // 5) == ((self.game_ctr - 2) // 5):
            if (state.action_wheel_angular_vel.left_back < 0 and new_state.action_wheel_angular_vel.left_back > 0) or \
            (state.action_wheel_angular_vel.left_back > 0 and new_state.action_wheel_angular_vel.left_back < 0):
                self.stucked_count += 1
                # print("count ", self.stucked_count, self.game_ctr)
        else:
            self.stucked_count = 0
        
        if self.stucked_count > 1:
            reward += -50 * self.stucked_count
            # print("count ", self.stucked_count)
        
        
        # print("total ", reward)
        # print("----------------------")

        return reward
    
    def step(self, state, new_state):
        self.episode_ctr += 1
        self.game_ctr += 1
        self.total_ctr += 1
        
        reward = self.calculate_reward(state, new_state)
        
        done, reachGoal = self.check_termination(state) #self.trailOrientation

        if reachGoal:
            reward += 1000
        
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
        # replay_buffer_size, try to fix

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
        feature.append(state.car_pos.x - state.final_target_pos.x) 
        feature.append(state.car_pos.y - state.final_target_pos.y)
        
        # angle in radian between up(0, 1)vector and car to target
        rad = Utility.radFromUp([state.car_pos.x, state.car_pos.y], \
            [state.final_target_pos.x, state.final_target_pos.y])
        feature.append(Utility.decomposeCosSin(rad)) #cos(radian), sin(radian) *2
        # print("car to target: ", rad / DEG2RAD)
        
        # car orientation(eular angles in radians)
        feature.append(Utility.decomposeCosSin(state.car_orientation)) 
        # print("car: ", state.car_orientation / DEG2RAD)
        
        # car velocity
        feature.append(state.car_vel.x)  
        feature.append(state.car_vel.y)
        # print(state.car_vel.x)

        # car angular velocity in radians(eular angles in radians)
        feature.append(state.car_angular_vel)
        # print(state.car_angular_vel)

        # 前進軸 angular velocity in radians *4 --> *1
        feature.append(state.wheel_angular_vel.left_back)
        feature.append(state.wheel_angular_vel.left_front)
        feature.append(state.wheel_angular_vel.right_back)
        feature.append(state.wheel_angular_vel.right_front)
        

        # min lidar displacement in meters
        # feature.append(-state.min_lidar_direciton.x)
        # feature.append(-state.min_lidar_direciton.y)

        # min lidar distance
        # feature.append(state.min_lidar)
        # feature.append(-1)

        # min lidar relative angle to car in radian
        # feature.append(Utility.decomposeCosSin(state.min_lidar_relative_angle))
        # feature.append(0)
        # feature.append(0)

        # angle in radian between up(0, 1)vector and car to obstacle
        # rad = Utility.radFromUp([state.car_pos.x, state.car_pos.y], \
        #     [state.car_pos.x + state.min_lidar_direciton.x, state.car_pos.y + state.min_lidar_direciton.y])
        # feature.append(Utility.decomposeCosSin(rad))

        # action
        feature.append(state.action_wheel_angular_vel.left_back)
        feature.append(state.action_wheel_angular_vel.left_front)
        feature.append(state.action_wheel_angular_vel.right_back)
        feature.append(state.action_wheel_angular_vel.right_front)

        
        feature = Utility.flatten(feature)
        # print(feature)
        return feature

def main(mode):
    print('The mode is:', mode)

    # TODO paramaterization
    # server = Server(port=5055) #5055
    # # t = CustomThread(server)
    env = Env(max_times_in_episode=10, max_times_in_game=210, end_distance=(0.2, 7), stop_target=False, target_fixed_sec=12)
  
    # 0518_car_to_target_few_features 0517_car_to_target_few_features
    chpt_dir_load = os.path.join(os.path.dirname(__file__),  'Model', 'DDPG', '1030_car/model') #0623_car_to_target_slow_retrain_double_prev_wheel_d_05 0621_car_to_target_slow_retrain_double_prev_wheel_d_05 0613_car_to_target_slow_retrain_double_prev:5000 0601_car_to_target_test_1
    chpt_dir_save = os.path.join(os.path.dirname(__file__),  'Model', 'DDPG', '1107_car/model')
    chpt_dir_plot = os.path.join(os.path.dirname(__file__),  'Model', 'DDPG', '1107_car/')
    chpt_dir_log  = os.path.join(os.path.dirname(__file__),  'Model', 'DDPG', '1107_car/log')
    # chpt_dir_buffer = os.path.join(os.path.dirname(__file__), '..', '..', 'Model', 'DDPG', '0709_car_to_target_slow_retrain_double_prev_/buffer')

    agent = Agt(q_lr=0.001, pi_lr=0.001, gamma=0.99, rho=0.005,  \
        pretrained=False, new_input_dims=17, \
        input_dims=17, n_actions=4, batch_size=100, layer1_size=400, layer2_size=300, \
        
        chpt_dir_load=chpt_dir_load, chpt_dir_save=chpt_dir_save)
    # replay_buffer_size=1000000, !! test\
    epoch = 5000

    reward_history, reward_history_ = ([] for i in range(2))

    unity_adaptor = UnityAdaptor(action_range=600, steering_angle_range=20)

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
                min_lidar=[],
                min_lidar_position=Entity.ROS2Point(x=0.0, y=0.0, z=0.0),
                second_min_lidar_position=Entity.ROS2Point(x=0.0, y=0.0, z=0.0),
                third_min_lidar_position=Entity.ROS2Point(x=0.0, y=0.0, z=0.0),
                max_lidar=0.0,
                min_lidar_direciton = [0.0],
                # min_lidar_direciton=Entity.ROS2Point(x=0.0, y=0.0, z=0.0),
                # max_lidar_position=Entity.ROS2Point(x=0.0, y=0.0, z=0.0),
                # min_lidar_relative_angle=0.0,
                action_wheel_angular_vel=Entity.WheelAngularVel(left_back=0.0, left_front=0.0, right_back=0.0, right_front=0.0),
                action_wheel_orientation=Entity.WheelOrientation(left_front=0.0, right_front=0.0))
    
    rclpy.init()
    node = AiNode()
    pros = threading.Thread(target=spin_props, args=(node,))
    pros.start()  

    try:
        if mode == 'train':
            lr_c_history, lr_a_history, critic_loss_history, actor_loss_history, critic_loss_history_, actor_loss_history_ = (
                [] for i in range(6))
            
            unity_obs = None
            
            # TODO paramaterization
            load_step = 2200  # 0
            agent.load_models(load_step)

            # if os.path.exists(chpt_dir_buffer):
            #     state_mem, action_mem, reward_mem, new_state_mem, terminal_mem = Utility.load_buffer(chpt_dir_buffer, load_step)
            #     agent.memory.store_buffer(state_mem, action_mem, reward_mem, new_state_mem, terminal_mem)

            # elapsed_time = 0.5  # s 0.06
            # prev_time_step = datetime.now().microsecond/1000000 - elapsed_time

            prev_pos = [0, 0]
            trail_original_pos = [0, 0]
            unity_action = [0, 0]

            for i in range(load_step+1, load_step+epoch+1):
                if unity_obs == None:
                    while unity_obs is None:
                        unity_obs = returnUnityState()
                    state = unity_adaptor.transfer_obs(unity_obs, unity_action)    
                    env.restart_game(state)
                else:
                    restart_game = env.restart_episode()
                    if restart_game:
                        # new_target = {'title': 'new target', 'content': {}} 
                        new_target = [1.0]
                        node.publish2Ros(new_target)
                        # server.sendAction(new_target)
                        # unity_obs = server.recvData()
                        unity_obs = returnUnityState()
                        state = unity_adaptor.transfer_obs(unity_obs, unity_action)    
                        env.restart_game(state)

                done = False
                score, lr_c, lr_a = (0 for i in range(3))
                c_loss, a_loss = ([] for i in range(2))

                while (not done):
                    ai_action = agent.choose_actions(state, prev_pos, trail_original_pos, inference=False)

                    action_sent_to_unity, unity_action = unity_adaptor.trasfer_action(ai_action)
                    node.publish2Ros(action_sent_to_unity)
                    time.sleep(0.5) ######

                    # calculate time
                    # timeStep = datetime.now().microsecond/1000000 #seconds
                    # elapsed_time = timeStep - prev_time_step
                    # if elapsed_time < 0:
                    #     elapsed_time += 1
                    # print(elapsed_time)
                    # prev_time_step = datetime.now().microsecond/1000000

                    unity_new_obs = returnUnityState()
                    new_state = unity_adaptor.transfer_obs(unity_new_obs, unity_action)
                    
                    reward, done, info = env.step(state, new_state)
                    
                    score += reward

                    agent.store_transition(state, ai_action, reward, new_state, int(done), prev_pos, trail_original_pos)

                    prev_pos = info['prev pos']
                    trail_original_pos = info['trail original pos']

                    state = new_state

                c_loss_episode = 0
                a_loss_episode = 0
                for j in range(env.episode_ctr):
                    c_loss_, a_loss_, lr_c, lr_a = agent.learn()
                    c_loss.append(c_loss_)
                    c_loss_episode += c_loss_
                    a_loss.append(a_loss_)
                    a_loss_episode += a_loss_

                reward_history.append(score)
                lr_c_history.append(lr_c)
                lr_a_history.append(lr_a)
                critic_loss_history.append(Utility.mean(c_loss))
                actor_loss_history.append(Utility.mean(a_loss))
                critic_loss_history_.append(c_loss_episode)
                actor_loss_history_.append(a_loss_episode)
                reward_history_.append(np.mean(reward_history[-50:]))

                if (i) % 100 == 0:
                    agent.save_models(i)
                    Utility.plot(reward_history_, lr_c_history, lr_a_history, critic_loss_history,
                                actor_loss_history, i, path=chpt_dir_plot)
                    # Utility.save_training_log(reward_history, critic_loss_history_, actor_loss_history_, chpt_dir_log)
                    # state_mem, action_mem, reward_mem, new_state_mem, terminal_mem = agent.memory.get_buffer()
                    # Utility.save_replay_buffer(state_mem, action_mem, reward_mem, new_state_mem, terminal_mem, chpt_dir_buffer, i)
                    
                print('episode:', i,
                    ',reward:%.2f' % (score),
                    ',avg reward:%.2f' % (reward_history_[-1]),
                    ',critic loss:%.2f' % (critic_loss_history[-1]),
                    ',actor loss:%.2f' % (actor_loss_history[-1]),
                    ',ctr:', env.episode_ctr,)

        elif mode == 'test':
            # agent.load_models(9000)
            agent.eval()
            unity_obs = None
            prev_pos = [0, 0]
            trail_original_pos = [0, 0]
            ai_action = [0, 0]

            # stucked_file = open(os.path.join(os.path.dirname(__file__), '..', '..', 'Model', 'DDPG', '0623_car_to_target_slow_retrain_double_prev_wheel_d_05', 'stucked.txt'), 'a')
            # state_file = open(os.path.join(os.path.dirname(__file__), '..', '..', 'Model', 'DDPG', '0623_car_to_target_slow_retrain_double_prev_wheel_d_05', 'unity_states.txt'), 'a')

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

                    action_sent_to_unity, unity_action = unity_adaptor.trasfer_action(ai_action)
                    # print(action_sent_to_unity)
                    server.sendAction(action_sent_to_unity)
                    time.sleep(0.2)
                    unity_new_obs = server.recvData()
                    new_state = unity_adaptor.transfer_obs(unity_new_obs, unity_action)
                    reward, done, info = env.step(state, new_state)

                    # data = Utility.count_stucked_times(new_state, env.stucked_count, env.game_ctr)
                    # stucked_file.write(data)
                    
                    # json_data = json.dumps(state.dict())
                    # state_file.write(json_data)

                    score += reward
                    prev_pos = info['prev pos']
                    trail_original_pos = info['trail original pos']

                    state = new_state

                print('test run: ', i, ',reward: %.2f,' % (score), 'ctr: ', env.game_ctr)

        server.close()

    except KeyboardInterrupt:
        server.close()
        # stucked_file.close()
        # state_file.close()

if __name__ == '__main__':
    #TODO paramaterization
    # mode = 'train'
    mode = 'train'
    main(mode)
