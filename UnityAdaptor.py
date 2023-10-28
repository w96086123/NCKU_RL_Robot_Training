from Entity import State
from Entity import ROS2Point
from Entity import WheelOrientation
from Entity import WheelAngularVel
from Entity import ControlSignal
import math
import numpy as np
from Utility import clamp

DEG2RAD = 0.01745329251
# FRONTDEGREE = 90

class UnityAdaptor():
    def __init__(self, action_range, steering_angle_range):
        self.action_range = action_range
        self.steering_angle_range = steering_angle_range
        self.prev_car_yaw = 0
    
    # Unity                    ROS2
    # 45 Eular angle(1, 1)     135(-1, 1)
    # 315           (-1, 1)    225(-1, -1)
    # def transfer_orientation(self, unity_orientation):
    #     ros2_orientation = math.pi - unity_orientation
    #     if ros2_orientation < 0:
    #         ros2_orientation += 2*math.pi
    #     if ros2_orientation == 2*math.pi:
    #         ros2_orientation = 0.0
    #     # print("ori")
    #     # print(unity_orientation / DEG2RAD, ros2_orientation / DEG2RAD)
    #     return ros2_orientation
    
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

        # obs['final target pos']['x']                             Unity coordinate
        #                          ['y']       

        # obs['car pos']['x']                                      Unity coordinate
        #                 ['y']                                      

        # obs['short time target pos']['closest point x']          Unity coordinate
        #                               ['closest point y']
        #                               ['second closest point x']
        #                               ['second closest point y']
        #                               ['farthest point x']
        #                               ['farthest point y']

        # obs['car velocity']['x']                                 Unity coordinate, m/s
        #                      ['y']                         

        # obs['car orientation']['value']                          Unity y rotation in radian, clockwise: 0 - 359 

        # obs['wheel orientation']['left back']                    Unity y rotation in radian, clockwise: 0 - 359 
        #                           ['left front']  
        #                           ['right back']
        #                           ['right front']
        
        # obs['car angular velocity']                              car y rotation speed, radian/s, clockwise: +

        # obs['wheel angular velocity']['left back']               car wheel rotation speed, radain/s, front: +, back: -
        #                                ['left front']      
        #                                ['right back']
        #                                ['right front']

        # obs['min range']['value']                                min lidar range in meters

        
        #     ROS2                                      Unity           
        # left(+ ++)    0   right(- ++)                       0
        #               x                           z
        #               ^                           ^
        #               |                           |
        #         y <---                             ---> x
        #              180
        # (forward)    ROS2_X = Unity_Z
        # (left/right) ROS2_Y = -Unity_X
        # (up)         ROS2_Z = Unity_Y 

        car_quaternion = [obs['ROS2CarQuaternion']['x'], obs['ROS2CarQuaternion']['y'],
                          obs['ROS2CarQuaternion']['z'], obs['ROS2CarQuaternion']['w']]
        car_roll_x, car_pitch_y, car_yaw_z = self.euler_from_quaternion(car_quaternion)
        car_orientation = self.radToPositiveRad(car_yaw_z)

        wheel_quaternion_left_front = [obs['ROS2WheelQuaternionLeftFront']['x'], obs['ROS2WheelQuaternionLeftFront']['y'], 
                                       obs['ROS2WheelQuaternionLeftFront']['z'], obs['ROS2WheelQuaternionLeftFront']['w']]
        wheel_left_front_roll_x, wheel_left_front_pitch_y, wheel_left_front_yaw_z = self.euler_from_quaternion(wheel_quaternion_left_front)
      
        wheel_quaternion_right_front = [obs['ROS2WheelQuaternionRightFront']['x'], obs['ROS2WheelQuaternionRightFront']['y'], 
                                        obs['ROS2WheelQuaternionRightFront']['z'], obs['ROS2WheelQuaternionRightFront']['w']]
        wheel_right_front_roll_x, wheel_right_front_pitch_y, wheel_right_front_yaw_z = self.euler_from_quaternion(wheel_quaternion_right_front)
    

        state = State(
            final_target_pos = ROS2Point(x = obs['ROS2TargetPosition']['x'], y = obs['ROS2TargetPosition']['y'], z=0.0),
            car_pos = ROS2Point(x = obs['ROS2CarPosition']['x'], y = obs['ROS2CarPosition']['y'], z=0.0),
            path_closest_pos = ROS2Point(x = obs['ROS2PathPositionClosest']['x'], y = obs['ROS2PathPositionClosest']['y'], z=0.0),
            path_second_pos = ROS2Point(x = obs['ROS2PathPositionSecondClosest']['x'], y = obs['ROS2PathPositionSecondClosest']['y'], z=0.0),
            path_farthest_pos = ROS2Point(x = obs['ROS2PathPositionFarthest']['x'], y = obs['ROS2PathPositionFarthest']['y'], z=0.0),
            car_vel = ROS2Point(x = obs['ROS2CarVelocity']['x'], y = obs['ROS2CarVelocity']['y'], z=0.0),
            car_orientation = car_orientation, 
            wheel_orientation = WheelOrientation(left_front = self.radToPositiveRad(wheel_left_front_yaw_z), \
                                                right_front = self.radToPositiveRad(wheel_right_front_yaw_z)),
            ##### angular v: unity world space 
            # Unity and ROS2 -> opposite direction
            car_angular_vel = obs['ROS2CarAugularVelocity']['z'],
            wheel_angular_vel = WheelAngularVel(left_back = obs['ROS2WheelAngularVelocityLeftBack']['y'], \
                                                left_front = obs['ROS2WheelAngularVelocityLeftFront']['y'], \
                                                right_back = obs['ROS2WheelAngularVelocityRightBack']['y'], \
                                                right_front = obs['ROS2WheelAngularVelocityRightFront']['y']
                                                ),
            min_lidar = obs['ROS2MinRange'],
            # min_lidar_direciton = ROS2Point(x = obs['ROS2MinRangeDirection']['x'], y = obs['ROS2MinRangeDirection']['y'], z=0.0),
            # min_lidar_relative_angle = self.calculate_relative_angle_unity_gloabl(obs['ROS2MinRangeDirection']['x'], obs['ROS2MinRangeDirection']['y'], 
            #                                                                       car_orientation / DEG2RAD),

            action_wheel_angular_vel = WheelAngularVel(left_back = self.discritize_wheel_angular_vel(ai_action[1]), \
                                                left_front = self.discritize_wheel_angular_vel(ai_action[1]), \
                                                right_back = self.discritize_wheel_angular_vel(ai_action[1]), \
                                                right_front = self.discritize_wheel_angular_vel(ai_action[1])
                                                ),
            action_wheel_orientation = WheelOrientation(left_front = self.discritize_wheel_steering_angle(ai_action[0]), \
                                                right_front = self.discritize_wheel_steering_angle(ai_action[0])),

            ##### angular v: unity local space
            # state.car_angular_vel = -obs['car angular velocity']['value']

            # state.wheel_angular_vel = WheelAngularVel()
            # state.wheel_angular_vel.left_back = -obs['wheel angular velocity']['left back']
            # state.wheel_angular_vel.left_front = -obs['wheel angular velocity']['left front']
            # state.wheel_angular_vel.right_back = -obs['wheel angular velocity']['right back']
            # state.wheel_angular_vel.right_front = -obs['wheel angular velocity']['right front']
        )
        # print("action in state")
        # print(ai_action)
        # print(self.discritize_wheel_steering_angle(ai_action[0]))
        # print(self.discritize_wheel_angular_vel(ai_action[1]))

        # print(f"car position: {state.car_pos.x:.2f}, {state.car_pos.y:.2f}")
        # print(f"closest path position: {state.path_closest_pos.x:.2f}, {state.path_closest_pos.y:.2f}")
        # print(f"furthest path position: {state.path_farthest_pos.x:.2f}, {state.path_farthest_pos.y:.2f}")
        # print(f"car velocity: {state.car_vel.x:.2f}, {state.car_vel.y:.2f}")
        # print(f"car orientation: {car_orientation/DEG2RAD:.2f}")
        # print(f"wheel orientation: {state.wheel_orientation.left_front/DEG2RAD:.2f}")
        # print(f"car yaw: {car_yaw_z/DEG2RAD}")
        # print(f"car prev yaw: {self.prev_car_yaw/DEG2RAD}")
        # print(f"car angular velocity: {state.car_angular_vel:.2f}, {((self.prev_car_yaw - car_yaw_z)/0.05):.2f}")
        # print(f"wheel angular velocity: {state.wheel_angular_vel.left_back:.2f}")
        
        # print(unity_action)
        # print(f"{state.action_wheel_orientation.left_front}, {state.action_wheel_angular_vel.left_back}")

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

        ############################# forward
        unity_action[1] = ai_action[1] * self.action_range 
        unity_action[1] = float(clamp(unity_action[1], -self.action_range, self.action_range))
        
        # print(unity_action)
        action_sent_to_unity = {'title': 'action', 'content': {'voltage': unity_action}}
   
        return action_sent_to_unity, unity_action
