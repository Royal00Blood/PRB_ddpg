import numpy as np
import torch
import torch.nn as  nn
import random
from collections import deque
from copy import deepcopy

class Noise():
    def __init__(self, action_dim, mu=0, theta=0.15, sigma=0.3):
        self.action_dim = action_dim
        self.mu = mu
        self.theta = theta
        self.sigma = sigma
        self.state = np.ones(self.action_dim) * self.mu
        self.reset()
    
    def reset(self):
        self.state = np.ones(self.action_dim) * self.mu
    
    def sample(self):
        x = self.state
        dx = self.theta * (self.mu - x) + self.sigma * np.random.randn(len(x))
        self.state = x + dx
        return self.state    

class Layers_NN(nn.Module):
    def __init__(self,input_dim, layer1_dim, layer2_dim, output_dim, output_tanh):
        super().__init__()
        self.layer1 = nn.Linear(input_dim,layer1_dim)
        self.layer2 = nn.Linear(layer1_dim,layer2_dim)
        self.layer3 = nn.Linear(layer2_dim,output_dim)
        self.output_tanh = output_tanh
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()
    
    def forward(self, input):
        hidden = self.layer1(input)
        hidden = self.relu(hidden)
        hidden = self.layer2(hidden)
        hidden = self.relu(hidden)
        output = self.layer3(hidden)
        
        if self.output_tanh:
            return self.tanh(output)
        else:
            return output

class DDPG():
    def __init__(self, state_dim, action_dim, action_scale, gamma=0.99, memory_size=1000000, q_lr=1e-3, pi_lr=1e-4, bath_size=64):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.action_scale = action_scale
        self.pi_model = Layers_NN(self.state_dim, 400, 300, self.action_dim, output_tanh=True)
        self.q_model = Layers_NN(self.state_dim + self.action_dim, 400, 300, 1, output_tanh=False)
        self.noise = Noise(self.action_dim)
        self.bath_size = bath_size 
        self.gamma = gamma
        self.memory = deque(maxlen=memory_size)
        self.q_optimazer = torch.optim.Adam(self.q_model.parameters(),lr=q_lr)
        self.pi_optimazer = torch.optim.Adam(self.pi_model.parameters(),lr=pi_lr)
        pass
    
    def get_action(self, state):
        print(state)
        state = torch.FloatTensor(state[0])
        pred_action = self.pi_model(state).detach().numpy()
        action = self.action_scale * (pred_action + self.noise.sample())
        return np.clip(action, -self.action_scale, self.action_scale)
    
    def fit(self, state, action, reward, done, next_state):
        self.memory.append([state, action, reward, done, next_state])
        
        if len(self.memory) > self.bath_size:
            batch = random.sample(self.memory, self.bath_size)
            
            states, actions, rewards, dones, next_states = [], [], [], [], []
            for el in batch:
                states.append(el[0])
                actions.append(el[1])
                rewards.append(el[2])
                dones.append(el[3])
                next_states.append(el[4])
        
            states = torch.FloatTensor(states)
            next_states = torch.FloatTensor(next_states)
            actions = torch.FloatTensor(actions)
                
            targets = []
            for i in range(self.bath_size):
                next_action = self.pi_model(next_states[i])
                
                next_state_and_action = torch.cat((next_states[i],next_action))
                target = rewards[i] + (1 - dones[i]) * self.gamma * self.q_model(next_state_and_action)
                targets.append(target)
            
            targets = torch.FloatTensor(targets)
            state_and_actions = torch.cat((states, actions),dim=1)
            q_loss = torch.mean((targets-self.q_model(state_and_actions))**2)
            q_loss.backward()
            self.q_optimazer.step()
            self.q_optimazer.zero_grad()
            
            pred_action = self.pi_model(states)
            states_and_pred_action = torch.cat((states, pred_action),dim=1)
            pi_loss = - torch.mean(self.q_model(states_and_pred_action))
            pi_loss.backward()
            self.pi_optimazer.step()
            self.pi_optimazer.zero_grad()
    
    
import gym

env = gym.make('Pendulum-v1')
agent = DDPG(state_dim=3,action_dim=1,action_scale=2)

episode_n = 200
trajectory_len = 200


for episode in range(episode_n):
    state = env.reset()
    total_reward = 0
    for _ in range(trajectory_len):
        action = agent.get_action(state)
        #print(env.step(action))
        next_action, reward, done, _, _ = env.step(action)
        next_action = next_action[0]
        agent.fit(state, action, reward, done, next_action)
        total_reward += reward
        
        if done:
            break
        state = next_action
        
    print(f"episode={episode}, total_reward={total_reward}")