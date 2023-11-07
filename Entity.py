from pydantic import BaseModel

class ROS2Point(BaseModel):
    x: float
    y: float
    z: float

# around ROS2 z axis, left +, right -, up 0, down 180
class WheelOrientation(BaseModel):
    left_front: float=0
    right_front: float=0

# around car wheel axis, front: +, back: -, r/s
class WheelAngularVel(BaseModel):
    left_back: float
    left_front: float
    right_back: float
    right_front: float

class State(BaseModel):
    final_target_pos: ROS2Point
    car_pos: ROS2Point
    # path: list = []
    path_closest_pos: ROS2Point
    path_second_pos: ROS2Point
    path_farthest_pos: ROS2Point
    car_vel: ROS2Point # in ROS2 coordinate system
    car_orientation: float # radians, around ROS2 z axis, counter-clockwise: 0 - 359
    wheel_orientation: WheelOrientation # around car z axis, counter-clockwise: +, clockwise: -, r/s
    car_angular_vel: float # r/s, in ROS2 around car z axis, yaw++: -, yaw--: +, counter-clockwise: +, clockwise: -, in Unity:  counter-clockwise: -, clockwise: +
    wheel_angular_vel: WheelAngularVel # around car wheel axis, front: +, back: -
    min_lidar: list # meter
    min_lidar_direciton: list

    # because orientation is transformed back to Unity coordinate system, here lidar direction alse needs to be transformed back from ROS2 to Unity
    # min_lidar_relative_angle: float # radian, base on car, right(x): 0, front(y): 90,  upper: 180 --->x 0, down: -180 --->x 0

    action_wheel_angular_vel: WheelAngularVel
    action_wheel_orientation: WheelOrientation

class ControlSignal(BaseModel):
    wheel_vel: float # rad/s
    steering_angle: float # degree, left: -, right: +