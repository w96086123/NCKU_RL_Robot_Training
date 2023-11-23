from Entity import State
from Entity import ROS2Point
from Entity import WheelOrientation
from Entity import WheelAngularVel
from Entity import ControlSignal
import math
import numpy as np
from Utility import clamp
import json

DEG2RAD = 0.01745329251
# FRONTDEGREE = 90

class UnityAdaptor():
    def __init__(self, action_range, steering_angle_range):
        self.action_range = action_range
        self.steering_angle_range = steering_angle_range
        self.prev_car_yaw = 0
    
    
    def euler_from_quaternion(self, orientation):
        """
        Convert a quaternion into euler angles (roll, pitch, yaw)
        roll is rotation around x in radians (counterclockwise)
        pitch is rotation around y in radians (counterclockwise)
        yaw is rotation around z in radians (counterclockwise)
        """
        x = orientation[0]
        y = orientation[1]
        z = orientation[2]
        w = orientation[3]

        t0 = +2.0 * (w * x + y * z)
        t1 = +1.0 - 2.0 * (x * x + y * y)
        roll_x = math.atan2(t0, t1)

        t2 = +2.0 * (w * y - z * x)
        t2 = +1.0 if t2 > +1.0 else t2
        t2 = -1.0 if t2 < -1.0 else t2
        pitch_y = math.asin(t2)

        t3 = +2.0 * (w * z + x * y)
        t4 = +1.0 - 2.0 * (y * y + z * z)
        yaw_z = math.atan2(t3, t4)

        return roll_x, pitch_y, yaw_z  # in radians
    
    # actually this transformation is back to Unity coordinate system...
    def radToPositiveRad(self, rad):
        # left +, right -, up 0, down 180 => clockwise: 0 - 359
        if rad < 0:
            rad = -rad
        elif rad > 0:
            rad = math.pi * 2 - rad

        return rad
    
    def discritize_wheel_steering_angle(self, action_wheel_steering_angle): #20~-20
        if action_wheel_steering_angle < 0:
            return -1
        elif action_wheel_steering_angle > 0:
            return 1
        else:
            return 0
    
    def discritize_wheel_angular_vel(self, action_wheel_angular_vel): #-1200~1200
        if action_wheel_angular_vel < 0:
            return -1
        elif action_wheel_angular_vel > 0:
            return 1
        else:
            return 0
    
    def angle_relative_to_x(self, dx, dy):
        return math.degrees(math.atan2(dy, dx))
    
    def calculate_relative_angle_unity_gloabl(self, lidar_direction_from_car_x, lidar_direction_from_car_y, car_orientation_degree_unity):
        relative_angle = self.angle_relative_to_x(-lidar_direction_from_car_y, lidar_direction_from_car_x)
        # print("original relative angle ", relative_angle)
        # print("car orientation ", self.carOrientation)
        relative_angle += car_orientation_degree_unity
        if relative_angle > 180:
            relative_angle -= 360
        if relative_angle <= -180:
            relative_angle += 360
        return relative_angle * DEG2RAD

    def transfer_obs(self, obs, ai_action):

        obs = json.loads(obs)
        
        for key, value in obs.items():
             if isinstance(value, str) and value.startswith('(') and value.endswith(')'):
                coordinate_str = value.strip('()')  
                coordinates = list(map(float, coordinate_str.split(',')))  
                obs[key] = coordinates  

        car_quaternion = [obs['ROS2CarQuaternion'][0], obs['ROS2CarQuaternion'][1],
                    obs['ROS2CarQuaternion'][2], obs['ROS2CarQuaternion'][3]]
        car_roll_x, car_pitch_y, car_yaw_z = self.euler_from_quaternion(car_quaternion)
        car_orientation = self.radToPositiveRad(car_yaw_z)

        wheel_quaternion_left_front = [obs['ROS2WheelQuaternionLeftFront'][0], 
                                       obs['ROS2WheelQuaternionLeftFront'][1], 
                                       obs['ROS2WheelQuaternionLeftFront'][2], 
                                       obs['ROS2WheelQuaternionLeftFront'][3]] #48 49 50 51ROS2WheelQuaternionRightBack
        wheel_left_front_roll_x, wheel_left_front_pitch_y, wheel_left_front_yaw_z = self.euler_from_quaternion(wheel_quaternion_left_front)
      
        wheel_quaternion_right_front = [obs['ROS2WheelQuaternionRightFront'][0], 
                                        obs['ROS2WheelQuaternionRightFront'][1], 
                                        obs['ROS2WheelQuaternionRightFront'][2], 
                                        obs['ROS2WheelQuaternionRightFront'][3]]
        wheel_right_front_roll_x, wheel_right_front_pitch_y, wheel_right_front_yaw_z = self.euler_from_quaternion(wheel_quaternion_right_front)
        
        # 將字串列表轉換為浮點數列表，如果 obs['ROS2Range'] 為空，則補足 36 個 0
        min_lidar = [float(value) if value else 0.0 for value in obs['ROS2Range']]

        # 補足 0 到長度為 36
        min_lidar += [0.0] * (36 - len(min_lidar))

        state = State(
            final_target_pos = ROS2Point(x = obs['ROS2TargetPosition'][0], 
                                         y = obs['ROS2TargetPosition'][1], 
                                         z=0.0),#obs[5]
            car_pos = ROS2Point(x = obs['ROS2CarPosition'][0], 
                                y = obs['ROS2CarPosition'][1], 
                                z=0.0),#obs[2]
            path_closest_pos = ROS2Point(x = obs['ROS2PathPositionClosest'][0], 
                                         y = obs['ROS2PathPositionClosest'][1], 
                                         z=0.0),#obs[8]
            path_second_pos = ROS2Point(x = obs['ROS2PathPositionSecondClosest'][0], 
                                        y = obs['ROS2PathPositionSecondClosest'][1], 
                                        z=0.0),#obs[11]
            path_farthest_pos = ROS2Point(x = obs['ROS2PathPositionFarthest'][0], 
                                          y = obs['ROS2PathPositionFarthest'][1], 
                                          z=0.0),#obs[14] #obs[15~17] ROS2CarPosition 
            car_vel = ROS2Point(x = obs['ROS2CarVelocity'][0], 
                                y = obs['ROS2CarVelocity'][1], 
                                z=0.0), #obs[20]
            car_orientation = car_orientation, 
            wheel_orientation = WheelOrientation(left_front = self.radToPositiveRad(wheel_left_front_yaw_z), \
                                                right_front = self.radToPositiveRad(wheel_right_front_yaw_z)),

            car_angular_vel = obs['ROS2CarAugularVelocity'][2], #obs[21 22]
            #obs[28]
            wheel_angular_vel = WheelAngularVel(left_back = obs['ROS2WheelAngularVelocityLeftBack'][1], #obs[30]                                               
                                                right_back = obs['ROS2WheelAngularVelocityRightBack'][1], #obs[34 36]
                                                ),
            stand_angular_vel = WheelAngularVel(left_front = obs['ROS2StandAngleLeft'][1],
                                                right_front = obs["ROS2StandAngleRight"][1]),
            min_lidar = min_lidar, #57 58 59
            min_lidar_direciton = [tuple(map(float, direction.strip('()').split(',')[:2])) for direction in obs["ROS2RangePosition"]],

            action_wheel_angular_vel = WheelAngularVel(left_back = self.discritize_wheel_angular_vel(ai_action[1]), \
                                                left_front = self.discritize_wheel_angular_vel(ai_action[1]), \
                                                right_back = self.discritize_wheel_angular_vel(ai_action[1]), \
                                                right_front = self.discritize_wheel_angular_vel(ai_action[1])
                                                ),
            action_wheel_orientation = WheelOrientation(left_front = self.discritize_wheel_steering_angle(ai_action[0]), \
                                                right_front = self.discritize_wheel_steering_angle(ai_action[0])),

            
        )
        

        self.prev_car_yaw = car_yaw_z

        
        return state
    
    def trasfer_action(self, ai_action):
        unity_action = [None, None]

        # print(ai_action)

        #### why cannot add negative?

        ############################# steering angle
        # TODO: Unity takes radian now, actually Unity can take degree
        #unity left: +, right: -
        unity_action[0] = ai_action[0] * self.steering_angle_range * DEG2RAD
        unity_action[0] = float(clamp(unity_action[0], -self.steering_angle_range  * DEG2RAD, self.steering_angle_range * DEG2RAD))

        # unity_action[0] = ai_action[0] * self.action_range 
        # unity_action[0] = float(clamp(unity_action[0], -self.action_range, self.action_range))

        ############################# forward
        unity_action[1] = ai_action[1] * self.action_range 
        unity_action[1] = float(clamp(unity_action[1], -self.action_range, self.action_range))
        
        # print(unity_action)
        action_sent_to_unity = [0.0, unity_action[0], unity_action[1]]
   
        return action_sent_to_unity, unity_action
