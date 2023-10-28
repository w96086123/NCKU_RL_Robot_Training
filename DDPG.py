import torch.nn as nn
import torch.optim as optim
import torch
import os
import numpy as np

class CriticNetwork(nn.Module):
    def __init__(self, q_lr, input_dims, fc1_dims, fc2_dims, n_actions, chept_dir_load, chept_dir_save, name):
        super(CriticNetwork, self).__init__()

        self.input_dims = input_dims
        self.n_actions = n_actions

        self.l1 = nn.Linear(input_dims + n_actions, fc1_dims)
        self.l2 = nn.Linear(fc1_dims, fc2_dims)
        self.l3 = nn.Linear(fc2_dims, 1)
        self.relu = nn.ReLU()

        self.name = name
        self.chept_dir_load = chept_dir_load
        self.chept_dir_save = chept_dir_save

        self.optimizer = optim.Adam(self.parameters(), lr=q_lr)
        # self.optimizer = optim.RMSprop(self.parameters(), lr=1e-4)
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, action):
        x = self.l1(torch.cat([state, action], 1))
        x = self.relu(x)
        x = self.l2(x)
        x = self.relu(x)
        x = self.l3(x)
        return x

    def save_checkpoint(self, tag):
        print('...saving checkpoint...')
        torch.save(self.state_dict(), os.path.join(
            self.chept_dir_save, self.name + '{t}' + '_ddpg').format(t=tag))

    def load_checkpoint(self, tag):
        print('...loading checkpoint...')
        self.load_state_dict(torch.load(os.path.join(
            self.chept_dir_load, 
            self.name + '{t}' + '_ddpg').format(t=tag), 
            map_location=self.device))
    
    def add_layers(self, new_input_dims):
        new_layer = nn.Linear(new_input_dims + self.n_actions, self.input_dims + self.n_actions)
        self.add_module('new_layer', new_layer)

class ActorNetwork(nn.Module):
    def __init__(self, pi_lr, input_dims, fc1_dims, fc2_dims, n_actions, chept_dir_load, chept_dir_save, name):
        super(ActorNetwork, self).__init__()

        self.input_dims = input_dims

        self.l1 = nn.Linear(input_dims, fc1_dims)
        self.l2 = nn.Linear(fc1_dims, fc2_dims)
        self.l3 = nn.Linear(fc2_dims, n_actions)
        self.relu = nn.ReLU()

        self.n_actions = n_actions
        self.name = name
        self.chept_dir_load = chept_dir_load
        self.chept_dir_save = chept_dir_save

        self.optimizer = optim.Adam(self.parameters(), lr=pi_lr)
        # self.optimizer = optim.RMSprop(self.parameters(), lr=1e-4)
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = self.l1(state)
        x = self.relu(x)
        x = self.l2(x)
        x = self.relu(x)
        x = self.l3(x)
        x = torch.tanh(x)
        return x

    def save_checkpoint(self, tag):
        print('...saving checkpoint...')
        torch.save(self.state_dict(), os.path.join(
            self.chept_dir_save, self.name + '{t}' + '_ddpg').format(t=tag))
    def load_checkpoint(self, tag):
        print('...loading checkpoint...')
        self.load_state_dict(torch.load(os.path.join(
            self.chept_dir_load, 
            self.name + '{t}' + '_ddpg').format(t=tag),
            map_location=self.device))
    
    def add_layers(self, new_input_dims):
        new_layer = nn.Linear(new_input_dims, self.input_dims)
        self.add_module('new_layer', new_layer)

class ReplayBuffer():
    def __init__(self, input_dims, n_actions, max_size = 1000000): ### 10000
        self.size = max_size
        self.cntr = 0
        self.state_mem = np.zeros((self.size, input_dims))
        self.action_mem = np.zeros((self.size, n_actions))
        self.reward_mem = np.zeros(self.size) 
        self.new_state_mem = np.zeros((self.size, input_dims))
        self.terminal_mem = np.zeros(self.size, dtype = np.float32)
    
    def store_transition(self, s, a, r, s_, d):
        index = self.cntr % self.size
        self.state_mem[index] = s
        self.action_mem[index] = a
        self.reward_mem[index] = r
        self.new_state_mem[index] = s_
        self.terminal_mem[index] = d
        self.cntr += 1
    
    def sample_buffer(self, batch_size):
        max_mem = min(self.cntr, self.size)
        # batch = np.random.choice(max_mem, batch_size)
        batch = np.random.randint(0, max_mem, size=batch_size)
        states = self.state_mem[batch]
        actions = self.action_mem[batch]
        rewards = self.reward_mem[batch]
        states_ = self.new_state_mem[batch]
        terminals = self.terminal_mem[batch]

        return states, actions, rewards, states_, terminals


class OUActionNoise(object):
    def __init__(self, mu, sigma=0.15, theta=.2, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(
                                                            self.mu, self.sigma)

class GaussianActionNoise(object):
    def __init__(self, mean: np.ndarray, sigma: np.ndarray):
        self._mu = mean
        self._sigma = sigma

    def __call__(self) -> np.ndarray:
        return np.random.normal(self._mu, self._sigma)

    def __repr__(self) -> str:
        return f"GaussianActionNoise(mu={self._mu}, sigma={self._sigma})"

class PretrainedCriticNetwork(nn.Module):
    def __init__(self, pretrained_module, q_lr, new_input_dims, original_input_dims, chept_dir_load, chept_dir_save, name):
        super(PretrainedCriticNetwork, self).__init__()

        self.l0 = nn.Linear(new_input_dims, original_input_dims)
        self.l1 = pretrained_module
        self.relu = nn.ReLU()

        self.name = name
        self.chept_dir_load = chept_dir_load
        self.chept_dir_save = chept_dir_save

        self.optimizer = optim.Adam(self.parameters(), lr=q_lr)
        # self.optimizer = optim.RMSprop(self.parameters(), lr=1e-4)
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state, action):
        x = self.l0(state)
        x = self.relu(x)
        x = self.l1(x, action)
        return x

    def save_checkpoint(self, tag):
        print('...saving checkpoint...')
        torch.save(self.state_dict(), os.path.join(
            self.chept_dir_save, self.name + '{t}' + '_ddpg').format(t=tag))

    def load_checkpoint(self, tag):
        print('...loading checkpoint...')
        self.load_state_dict(torch.load(os.path.join(
            self.chept_dir_load, 
            self.name + '{t}' + '_ddpg').format(t=tag), 
            map_location=self.device))

class PretrainedActorNetwork(nn.Module):
    def __init__(self, pretrained_module, pi_lr, new_input_dims, original_input_dims, chept_dir_load, chept_dir_save, name):
        super(PretrainedActorNetwork, self).__init__()

        self.l0 = nn.Linear(new_input_dims, original_input_dims)
        self.l1 = pretrained_module
        self.relu = nn.ReLU()

        self.name = name
        self.chept_dir_load = chept_dir_load
        self.chept_dir_save = chept_dir_save

        self.optimizer = optim.Adam(self.parameters(), lr=pi_lr)
        # self.optimizer = optim.RMSprop(self.parameters(), lr=1e-4)
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
        self.to(self.device)

    def forward(self, state):
        x = self.l0(state)
        x = self.relu(x)
        x = self.l1(x)
        return x

    def save_checkpoint(self, tag):
        print('...saving checkpoint...')
        torch.save(self.state_dict(), os.path.join(
            self.chept_dir_save, self.name + '{t}' + '_ddpg').format(t=tag))
    def load_checkpoint(self, tag):
        print('...loading checkpoint...')
        self.load_state_dict(torch.load(os.path.join(
            self.chept_dir_load, 
            self.name + '{t}' + '_ddpg').format(t=tag),
            map_location=self.device))