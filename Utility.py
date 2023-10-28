import matplotlib.pyplot as plt
import os
import numpy as np
import math

DEG2RAD = 0.01745329251

def clamp(n, smallest, largest):
    return max(smallest, min(n, largest))

def four2eight(action):
    action_cpy = [0,0,0,0]
    for i in range(4):
        action_cpy.append(float(action[i]))

    return action_cpy

def rad2deg(rad):
    return rad / DEG2RAD

# def radToPositiveDeg(self, rad):
#         # left +, right -, up 0, down 180 => clockwise: 0 - 359
#         deg = rad / DEG2RAD
#         if deg < 0:
#             deg = -deg
#         elif deg > 0:
#             deg = 360 - deg

#         return deg


#                         x
#                         ^
#                         |
# ROS2 coordinate y <-----|
# left +, right -, up 0, down 180
def radFromUp(pos, targetPos):
    v = np.array(targetPos) - np.array(pos)
    up = np.array([1,0])
    rad = angle_between(v, up)
    if (v[1] > 0):
        rad = 2 * math.pi - rad
    return rad

def angle_between(v1, v2): #in radians
    
    if (v1 == np.array([0.0, 0.0])).all():
        v1_u = np.array([0.0, 0.0])
    else:
        v1_u = unit_vector(v1)
    
    if (v2 == np.array([0.0, 0.0])).all():
        v2_u = np.array([0.0, 0.0])
    else:
        v2_u = unit_vector(v2)

    return np.arccos(np.clip(np.dot(v1_u, v2_u), -1.0, 1.0))

def unit_vector(vector):
    return vector / np.linalg.norm(vector)

def decomposeCosSin(angle):
    return [np.cos(angle), np.sin(angle)]

def flatten(list_of_lists):
    if len(list_of_lists) == 0:
        return list(list_of_lists)
    if hasattr(list_of_lists[0], '__iter__'):
        return flatten(list_of_lists[0]) + flatten(list_of_lists[1:])
    return list(list_of_lists[:1]) + flatten(list_of_lists[1:])

def mean(lst):
    return sum(lst) / len(lst)



def plot(reward, lr_c, lr_a, crtirc_loss, actor_loss, n_episodes, path, show = False):
    length = len(reward)

    x = [i for i in range(length)]
    
    figure, axis = plt.subplots(2,2)

    for r in reward:
        if r  < -1000:
            r = -1000
    axis[0,0].plot(x, reward)
    axis[0,0].set_title('Reward')
    axis[0,1].plot(x, lr_c)
    axis[0,1].plot(x, lr_a, color = 'green')
    axis[0,1].set_title('Learning_rate')
    axis[1,0].plot(x, crtirc_loss)
    axis[1,0].set_title('Crtic_loss')
    axis[1,1].plot(x, actor_loss)
    axis[1,1].set_title('Actor_loss')
   
    axis[1,0].set(xlabel='episodes')
    axis[1,1].set(xlabel='episodes')
    axis[0,0].label_outer()
    axis[0,1].label_outer()

    plt.savefig(os.path.join(path) + '/{n}_episodes.png'.format(n = n_episodes))
    if show:
        plt.show()

def plot_PPO(reward, entropy, crtirc_loss, actor_loss, n_episodes, path, show = False):
    length = len(reward)

    x = [i for i in range(length)]
    
    figure, axis = plt.subplots(2,2)

    axis[0,0].plot(x, reward)
    axis[0,0].set_title('Reward')
    axis[0,1].plot(x, entropy)
    # axis[0,1].plot(x, entropy, color = 'green')
    axis[0,1].set_title('Entropy')
    axis[1,0].plot(x, crtirc_loss)
    axis[1,0].set_title('Crtic_loss')
    axis[1,1].plot(x, actor_loss)
    axis[1,1].set_title('Actor_loss')
   
    axis[1,0].set(xlabel='episodes')
    axis[1,1].set(xlabel='episodes')
    axis[0,0].label_outer()
    axis[0,1].label_outer()

    plt.savefig(os.path.join(path) + '/{n}_episodes.png'.format(n = n_episodes))
    if show:
        plt.show()
