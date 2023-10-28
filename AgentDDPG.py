
import DDPG
import Utility
from stable_baselines3.common.noise import NormalActionNoise
import torch.nn.functional as F
import numpy as np
import torch
import abc
import os

class Agent():
    def __init__(self, q_lr, pi_lr, gamma, rho, pretrained=False, new_input_dims=12, input_dims=11, n_actions=4,
                 layer1_size=400, layer2_size=300, batch_size=100, chpt_dir_load='Model/DDPG', \
                    chpt_dir_save='Model/DDPG'):
        self.rho = rho
        self.gamma = gamma
        self.batch_size = batch_size
   
        # self.noice = ddpg.OUActionNoise(mu = np.zeros(n_actions)) don't know why the result is badbadbad
        # TODO: use DDPG Gaussian as NormalActionNoise  
        self.noice = NormalActionNoise(mean=np.zeros(
            n_actions), sigma=0.1 * np.ones(n_actions))
        self.device = torch.device(
            'cuda:0' if torch.cuda.is_available() else 'cpu')
        self.updates = 0
        self.pretrained = pretrained
        self.new_input_dims = new_input_dims

        if not os.path.exists(chpt_dir_save):
            os.makedirs(chpt_dir_save)

        if self.pretrained:
            self.memory = DDPG.ReplayBuffer(input_dims=new_input_dims, n_actions=n_actions)

            self.pretrained_critic = DDPG.CriticNetwork(q_lr, input_dims, layer1_size, layer2_size, n_actions, \
                chpt_dir_load, chpt_dir_save, name='Crirtic_')
            self.pretrained_actor = DDPG.ActorNetwork(pi_lr, input_dims, layer1_size, layer2_size, n_actions, \
                chpt_dir_load, chpt_dir_save, name='Actor_')
            self.pretrained_target_critic = DDPG.CriticNetwork(q_lr, input_dims, layer1_size, layer2_size, n_actions, \
                chpt_dir_load, chpt_dir_save, name='TargetCrirtic_')
            self.pretrained_target_critic.load_state_dict(self.pretrained_critic.state_dict())
            self.pretrained_target_actor = DDPG.ActorNetwork(pi_lr, input_dims, layer1_size, layer2_size, n_actions, \
                chpt_dir_load, chpt_dir_save, name='TargetActor_')
            self.pretrained_target_actor.load_state_dict(self.pretrained_actor.state_dict())
                                                        
            self.critic = DDPG.PretrainedCriticNetwork(self.pretrained_critic, q_lr, new_input_dims, input_dims, chpt_dir_load, chpt_dir_save, name='Crirtic_')
            self.actor = DDPG.PretrainedActorNetwork(self.pretrained_actor, pi_lr, new_input_dims, input_dims, chpt_dir_load, chpt_dir_save, name='Actor_')
            self.target_critic = DDPG.PretrainedCriticNetwork(self.pretrained_target_critic, q_lr, new_input_dims, input_dims, chpt_dir_load, chpt_dir_save, name='TargetCrirtic_')
            self.target_critic.load_state_dict(self.critic.state_dict())
            self.target_actor = DDPG.PretrainedActorNetwork(self.pretrained_target_actor, pi_lr, new_input_dims, input_dims, chpt_dir_load, chpt_dir_save, name='TargetActor_')
            self.target_actor.load_state_dict(self.actor.state_dict())
        else:
            self.memory = DDPG.ReplayBuffer(input_dims=input_dims, n_actions=n_actions)

            self.critic = DDPG.CriticNetwork(q_lr, input_dims, layer1_size, layer2_size, n_actions, \
                chpt_dir_load, chpt_dir_save, name='Crirtic_')
            self.actor = DDPG.ActorNetwork(pi_lr, input_dims, layer1_size, layer2_size, n_actions, \
                chpt_dir_load, chpt_dir_save, name='Actor_')
            self.target_critic = DDPG.CriticNetwork(q_lr, input_dims, layer1_size, layer2_size, n_actions, \
                chpt_dir_load, chpt_dir_save, name='TargetCrirtic_')
            self.target_critic.load_state_dict(self.critic.state_dict())
            self.target_actor = DDPG.ActorNetwork(pi_lr, input_dims, layer1_size, layer2_size, n_actions, \
                chpt_dir_load, chpt_dir_save, name='TargetActor_')
            self.target_actor.load_state_dict(self.actor.state_dict())

        self.update_network_parameters(rho=1)

        # self.critic_lr_scheduler = torch.optim.lr_scheduler.StepLR(self.critic.optimizer, 
        #                                         step_size = 50)
        # self.actor_lr_scheduler = torch.optim.lr_scheduler.StepLR(self.actor.optimizer, 
        #                                         step_size = 50)

    def update_network_parameters(self, rho=None):
        if rho is None:
            rho = self.rho
        # rho = self.rho
        else:
            print(rho)
        critic_params = self.critic.named_parameters()
        actor_params = self.actor.named_parameters()
        target_critic_params = self.target_critic.named_parameters()
        target_actor_params = self.target_actor.named_parameters()

        critic_params_dict = dict(critic_params)
        actor_params_dict = dict(actor_params)
        target_critic_params_dict = dict(target_critic_params)
        target_actor_params_dict = dict(target_actor_params)

        for name in critic_params_dict:
            critic_params_dict[name] = rho * critic_params_dict[name].clone() + \
                (1 - rho) * target_critic_params_dict[name].clone()
        self.target_critic.load_state_dict(critic_params_dict)

        for name in actor_params_dict:
            actor_params_dict[name] = rho * actor_params_dict[name].clone() + \
                (1 - rho) * target_actor_params_dict[name].clone()
        self.target_actor.load_state_dict(actor_params_dict)

    def learn(self):
        self.updates += 1
        critic_losses, actor_losses = 0, 0

        state, actions, reward, new_state, d = self.memory.sample_buffer(
            self.batch_size)
        state = torch.tensor(state, dtype=torch.float).to(self.device)
        actions = torch.tensor(actions, dtype=torch.float).to(self.device)
        reward = torch.tensor(reward, dtype=torch.float).to(self.device)
        new_state = torch.tensor(new_state, dtype=torch.float).to(self.device)
        d = torch.tensor(d).to(self.device)

        # if self.pretrained:
        #     for param in self.pretrained_critic.parameters():
        #         param.requires_grad = False
        # print("before no_grad", list(self.pretrained_critic.parameters())[-1])

        with torch.no_grad():
            # print("after no_grad", list(self.pretrained_critic.parameters())[-1])
            self.target_actor.eval()
            target_actions = self.target_actor.forward(new_state)
            self.target_critic.eval()
            critic_value_ = self.target_critic.forward(
                new_state, target_actions)  # batch size

            target = []
            for i in range(self.batch_size):
                target.append(reward[i] + (1 - d[i]) *
                              self.gamma * critic_value_[i])
            target = torch.tensor(target, dtype=torch.float).to(self.device)
            target = target.view(self.batch_size, 1)
            # target = reward + (1 - d) * self.gamma * critic_value_

        self.critic.eval()
        critic_value = self.critic.forward(state, actions)

        critic_loss = F.mse_loss(target, critic_value)
        critic_losses = critic_loss.item()

        self.critic.train()
        self.critic.optimizer.zero_grad()  # clean the previous grdient
        critic_loss.backward()  # claculate gradient
        self.critic.optimizer.step()  # update paramters

        lr_c = self.get_lr(self.critic.optimizer)

        # self.critic_lr_scheduler.step()
        # lr_c = self.critic_lr_scheduler.get_last_lr()

        if self.updates % 1 == 0:
            self.actor.eval()
            actions = self.actor.forward(state)

            self.critic.eval()
            actor_loss = -self.critic.forward(state, actions)
            actor_loss = torch.mean(actor_loss)
            actor_losses = actor_loss.item()

            self.actor.train()
            self.actor.optimizer.zero_grad()
            actor_loss.backward()
            self.actor.optimizer.step()

            lr_a = self.get_lr(self.actor.optimizer)

            # self.actor_lr_scheduler.step()
            # lr_a = self.actor_lr_scheduler.get_last_lr()

            self.update_network_parameters()

        return critic_losses, actor_losses, lr_c, lr_a

    def get_lr(self, optimizer):
        for param_group in optimizer.param_groups:
            return param_group['lr']

    def save_models(self, tag):
        self.critic.save_checkpoint(tag)
        self.actor.save_checkpoint(tag)
        self.target_critic.save_checkpoint(tag)
        self.target_actor.save_checkpoint(tag)

    def load_models(self, tag):
        if self.pretrained:
            self.pretrained_critic.load_checkpoint(tag)
            self.pretrained_actor.load_checkpoint(tag)
            self.pretrained_target_critic.load_checkpoint(tag)
            self.pretrained_target_actor.load_checkpoint(tag)
        else:
            self.critic.load_checkpoint(tag)
            self.actor.load_checkpoint(tag)
            self.target_critic.load_checkpoint(tag)
            self.target_actor.load_checkpoint(tag)
      
    def eval(self):
        self.critic.eval()
        self.actor.eval()
        self.target_critic.eval()
        self.target_actor.eval()
    
    def store_transition(self, obs, actions, rewards, new_obs, done, prev_pos, trail_original_pos):
        self.memory.store_transition(self.processFeature(obs, prev_pos, trail_original_pos), actions, rewards,
                                              self.processFeature(new_obs, prev_pos, trail_original_pos), done)

    # override
    def choose_actions(self, obs, prev_pos, trail_original_pos, inference: bool):
        pass
  
    
    # override
    def processFeature(self, state: dict, prev_pos, trail_original_pos):
        pass
 