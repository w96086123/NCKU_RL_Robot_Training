import torch
import Utility
import math
# from CustomThread import CustomThread
import os
import json
from datetime import datetime
from Entity import State



class Environment():
    def __init__(self, max_times_in_episode, max_times_in_game, end_distance=(1, 15), \
        save_log=False, stop_target=True, target_fixed_sec=8, min_angle_diff=10):
        self.devie = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.pos = [0, 0]
        self.target_pos = None
        self.real_target = None
        self.episode_ctr = 0
        self.game_ctr = 0
        self.total_ctr = 0
        self.prev_pos = [0, 0]
        self.orientation = None
        self.prevOrientation = 0

        self.prev_time = datetime.now().second
        self.time = datetime.now().second

        self.max_times_in_episode = max_times_in_episode
        self.max_times_in_game = max_times_in_game

        self.save_log = save_log
        if save_log:
            self.log = {"obs": []}
            with open(os.path.join(os.path.dirname(__file__), "state_log_virtual_edge.txt"), "w") as f:
                json.dump(self.log, f)
        
        self.target_fixed_sec = target_fixed_sec
        self.stop_target = stop_target
        self.end_distance = end_distance
        self.epsilon = 0.0001
        self.min_angle_diff = min_angle_diff

        self.distance_out = False
        self.game_finished = False
        self.reach_goal = False
        

    def calculate_orientation_diff(self, car_orientation, target_orientation):
        diff = abs(target_orientation - car_orientation)
        if diff > 180:
            diff = 360 - diff
        reward = diff

        return reward

    def calculate_distance(self, car_pos, target_pos):
        distance = math.dist(car_pos, target_pos)
        return distance

    # def radToPositiveDeg(self, rad):
    #     # left +, right -, up 0, down 180 => clockwise: 0 - 359
    #     deg = rad / DEG2RAD
    #     if deg < 0:
    #         deg = -deg
    #     elif deg > 0:
    #         deg = 360 - deg

    #     return deg

    def save_log(self):
        f = open(os.path.join(os.path.dirname(__file__), "state_log_virtual_edge.txt"), 'w')
        log = json.dumps((self.log))
        f.seek(0)
        f.write(log)
        f.truncate()
        f.close()

    def restart_episode(self):
        if (self.save_log):
            self.save_log()

        self.episode_ctr = 0

        restart_game = False

        if (self.distance_out == True or self.reach_goal == True or self.game_finished == True):
            if self.distance_out == True:
                print("distance is out")
            if self.reach_goal == True:
                print("reaches goal")
            if self.game_finished == True:
                print("game is finished")
            restart_game = True

        if (self.stop_target == True):
            self.game_ctr = 0
            if (self.time < self.prev_time):
                self.prev_time -= 60
            if (self.time - self.prev_time) >= self.target_fixed_sec:
                self.prev_time = self.time
            restart_game = True    
                    
        return restart_game

    def restart_game(self, state: State):
        self.init = 1
        self.game_ctr = 0
        self.pos = [state.car_pos.x, state.car_pos.y]
        self.inital_pos = self.pos
        # self.pos = [0., 0.] 
    
        self.target_pos = [state.final_target_pos.x, state.final_target_pos.y]
        # self.target_pos = [self.pos[0] + obs['final target pos']['x'], self.pos[1] + obs['final target pos']['y']]

        self.trail_original_pos = [state.path_closest_pos.x, state.path_closest_pos.y]
        self.distance = self.calculate_distance(self.pos, self.target_pos)

    # override
    def check_termination(self):
        pass
     
    # override
    def step(self, obs, new_obs):
        pass
      