import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from models.actors import Actor_1
from models.critics import Critic1, Critic2
import os
from settings import (STATE_SIZE, ACTION_SIZE, LR_ACTOR,
                      LR_CRITIC,BATCH_SIZE,GAMMA,BUFFER_SIZE,
                      ALPHA,TAU,EPISODES,EP_STEPS,TEST_EP_STEPS,
                      TEST_EPISODES, ACTION_,WEIGHT_DEC)
from torch.utils.tensorboard import SummaryWriter
from torchrl.data import PrioritizedReplayBuffer
from buffers.PrioritizedReplayBuffer import PrioritizedReplayBuffer
import time
import torch.nn.functional as F
from noise import Noise
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class PRB_DDPG_Agent:
    def __init__(self, 
                 state_size=STATE_SIZE, 
                 action_size=ACTION_SIZE,
                 lr_actor=LR_ACTOR, 
                 lr_critic=LR_CRITIC, 
                 gamma=GAMMA,tau=TAU, 
                 buffer_size=BUFFER_SIZE, 
                 batch_size=BATCH_SIZE, 
                 alpha=ALPHA,
                 weight_decay = WEIGHT_DEC
                 ):
        self.state_size = state_size
        self.action_size = action_size
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.gamma = gamma
        self.tau = tau
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.action_scale = ACTION_
        self.replay_buffer = PrioritizedReplayBuffer(buffer_size, alpha)
        self.writer = SummaryWriter(comment=f"lr_actor={lr_actor}, lr_critic={lr_critic}, gamma={gamma}, tau={tau}, buffer_size={buffer_size}, batch_size={batch_size}")
        self.global_step = 0
        self.noise = Noise(self.action_size)
        self.weight_decay = weight_decay
        
        self.actor = torch.compile(Actor_1(state_size, action_size)).to(device)
        self.critic1 = torch.compile(Critic1(state_size + action_size)).to(device)
        self.critic2 = torch.compile(Critic2(state_size + action_size)).to(device)
        self.actor_target = torch.compile(Actor_1(state_size, action_size)).to(device)
        self.critic1_target = torch.compile(Critic1(state_size + action_size)).to(device)
        self.critic2_target = torch.compile(Critic2(state_size + action_size)).to(device)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr_actor,weight_decay=self.weight_decay)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=self.lr_critic,weight_decay=self.weight_decay)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=self.lr_critic,weight_decay=self.weight_decay)  # Оптимизатор для второго критика
        
    def get_action(self,state):
        state = torch.tensor(state, dtype=torch.float32).to(device)
        action = self.action_scale * (self.actor(state).detach().cpu().numpy() + self.noise.sample())
        return np.clip(action, -self.action_scale, self.action_scale)
    
    def soft_update(self, source, target, tau):
        """Softly update target network parameters."""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
            
    def update_target_networks(self):
        """Update the target networks."""
        self.soft_update(self.actor, self.actor_target, self.tau)
        self.soft_update(self.critic1, self.critic1_target, self.tau)
        self.soft_update(self.critic2, self.critic2_target, self.tau)
    
    def update(self):
        transitions, indices, weights = self.replay_buffer.sample(self.batch_size)
        if transitions is None:
            return
        states, actions, rewards, next_states, dones = zip(*transitions)
        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(np.array(actions), dtype=torch.float32)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32)
        dones = torch.tensor(np.array(dones), dtype=torch.float32)
        weights = torch.tensor(np.array(weights), dtype=torch.float32)
        
        states = states.to(device)
        actions = actions.to(device)
        next_states = next_states.to(device)
        rewards = rewards.to(device)
        dones = dones.to(device)
        weights = weights.to(device)

        # Update Critic
        next_states_action = torch.cat((next_states, self.actor_target(next_states)),dim=1)
        next_q_values1 = self.critic1_target(next_states_action)
        next_q_values2 = self.critic2_target(next_states_action)
        target_q_values=[]
        for i in range(len(rewards)):
            target_q_values.append(rewards[i] + self.gamma * (1 - dones[i]) * torch.min(next_q_values1, next_q_values2)[i])  # Усреднение ошибок
        target_q_values = torch.tensor(target_q_values).to(device)
        ###
        states_action = torch.cat((states, actions), dim=1).to(device)
        critic1_loss = (weights * nn.MSELoss(reduction='none')(self.critic1(states_action), target_q_values.detach().unsqueeze(1))).mean()
        critic2_loss = (weights * nn.MSELoss(reduction='none')(self.critic2(states_action), target_q_values.detach().unsqueeze(1))).mean()
        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()
        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()
        # Update Actor
        states_a_actor = torch.cat((states, self.actor(states)), dim=1)
        actor_loss = -(weights * self.critic1(states_a_actor)).mean()  # Используем первый критик для обновления актера
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        # Update Priorities
        new_priorities = (critic1_loss.detach().cpu().numpy() + critic2_loss.detach().cpu().numpy() + 1e-5) / 2
        self.replay_buffer.update_priority(indices, new_priorities)
        # Update Target Networks
        self.update_target_networks()
        
        self.writer.add_scalar('Actor Loss'  , actor_loss.item()  , self.global_step)
        self.writer.add_scalar('Critic1 Loss', critic1_loss.item(), self.global_step)
        self.writer.add_scalar('Critic2 Loss', critic2_loss.item(), self.global_step)
        #self.writer.add_scalar('Average Reward', torch.mean(rewards), self.global_step)
        self.global_step += 1
       
        
    def train(self, env, num_episodes=EPISODES, ep_steps=EP_STEPS):
        #Загрузка весов и моделей для продолжения обучения
        # if os.path.exists('actor_weights.pth') and os.path.exists('critic_weights.pth'):
        #     self.actor.load_state_dict(torch.load('actor_weights.pth'))
        #     self.critic1.load_state_dict(torch.load('critic1_weights.pth'))
        #     self.critic2.load_state_dict(torch.load('critic2_weights.pth'))
        
        start_time = time.time()
        episode_rewards = []
        
        for episode in range(num_episodes):
            state = env.reset_env()
            done = False
            episode_reward = 0
            env.set_number(episode)
            i=0
            
            while not done:
                action =self.get_action(state) 
                next_state, reward, done, _ = env.step(action)
                
                if i>150:
                    reward-=500
                    break
                i+=1
                
                self.replay_buffer.push(state, action, reward, next_state, done)
                if len(self.replay_buffer) > self.batch_size:
                    self.update()
                state = next_state
                episode_reward += reward
            episode_rewards.append(episode_reward)
            avg_reward = np.mean(episode_rewards[-100:])
            
            print(f"Episode {episode+1}/{num_episodes}, Reward: {episode_reward:.2f}, Avg Reward: {avg_reward:.2f}")

            if episode % 5 == 0:
                torch.save(self.actor.state_dict()  , 'actor_weights.pth')
                torch.save(self.critic1.state_dict(), 'critic1_weights.pth')
                torch.save(self.critic2.state_dict(), 'critic2_weights.pth')
            
        # Сохранение весов и моделей
        torch.save(self.actor.state_dict()  ,'actor_weights.pth')
        torch.save(self.critic1.state_dict(),  'critic1_weights.pth')
        torch.save(self.critic2.state_dict(),  'critic2_weights.pth')
        
        self.writer.close()
        end_time = time.time()
        print(f"Training time: {end_time - start_time:.2f} seconds")
        return episode_rewards

    def test(self, env, max_episodes=TEST_EPISODES, max_steps=TEST_EP_STEPS):
        """
        Функция для тестирования обученной модели DDPG.
        
        Args:
            env (object): пользовательская среда
            actor_net (torch.nn.Module): обученная модель актора
            max_episodes (int): максимальное количество эпизодов для тестирования
            max_steps (int): максимальное количество шагов в одном эпизоде
        """
        # Загрузка весов и моделей
        
        self.actor.load_state_dict(torch.load( 'actor_weights.pth'))
        self.critic1.load_state_dict(torch.load('critic1_weights.pth'))
        self.critic2.load_state_dict(torch.load( 'critic2_weights.pth'))
       
        rewards = []
       
        for episode in range(max_episodes):
            state = env.reset_env()
            total_reward = 0
            for step in range(max_steps):
                # Выбираем действие на основе текущего состояния
                action = self.get_action(state)
                # Применяем действие в среде и получаем новое состояние, награду и флаг завершения
                next_state, reward, done, _ = env.step(action)
                total_reward += reward
                if done:
                    break
                state = next_state
                
            rewards.append(total_reward)
            print(f"Episode {episode+1}, Reward: {total_reward:.2f}")
        
        print(f"Average reward: {np.mean(rewards):.2f}")
