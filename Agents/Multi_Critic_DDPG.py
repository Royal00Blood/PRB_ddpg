import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from models.actors import Actor
from models.critics import Critic
import os
from settings import (S_SIZE, A_SIZE, LR_A, LR_C, L_C1, L_C2,
                      BATCH_SIZE,GAMMA,BUFFER_SIZE,
                      ALPHA, TAU, EPISODES,
                      EP_STEPS, TEST_EP_STEPS,
                      TEST_EPISODES, A_MAX, WEIGHT_DEC)
from torch.utils.tensorboard import SummaryWriter
from buffers.PrioritizedReplayBuffer import PrioritizedReplayBuffer
import time
import torch.nn.functional as F
from noise import Noise
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# path = "C:\Users\Ivan\Documents\python_github\PRB_ddpg\Agents\models_save"

class PRB_DDPG_Agent:
    def __init__(self, state_size=S_SIZE, action_size=A_SIZE, lr_actor=LR_A, lr_critic=LR_C, 
                 gamma=GAMMA, tau=TAU, buffer_size=BUFFER_SIZE, batch_size=BATCH_SIZE, alpha=ALPHA,
                 weight_decay = WEIGHT_DEC, action_max=A_MAX):
        
        self.state_size = state_size
        self.action_size = action_size
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.gamma = gamma
        self.tau = tau
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.action_max = action_max 
        self.weight_decay = weight_decay
        self.replay_buffer = PrioritizedReplayBuffer(buffer_size, alpha)
        comment = f"lr_actor={lr_actor}, lr_critic={lr_critic}, gamma={gamma}, tau={tau}, buffer_size={buffer_size}, batch_size={batch_size}"
        self.writer = SummaryWriter(comment=comment)
        self.global_step = 0
        self.noise = Noise(self.action_size)
        self.layers_critic_1 = L_C1
        self.layers_critic_2 = L_C2
        
        self.actor = torch.compile(Actor(self.state_size, self.action_size)).to(device)
        self.critic1 = torch.compile(Critic(self.state_size, self.action_size,layers=self.layers_critic_1, name="critic1")).to(device)
        self.critic2 = torch.compile(Critic(self.state_size, self.action_size,layers=self.layers_critic_2, name="critic2")).to(device)
        
        self.actor_target = torch.compile(Actor(state_size, action_size,name="actor_target")).to(device)
        self.critic1_target = torch.compile(Critic(self.state_size, self.action_size,layers=self.layers_critic_1,name="critic1_target")).to(device)
        self.critic2_target = torch.compile(Critic(self.state_size, self.action_size,layers=self.layers_critic_2,name="critic2_target")).to(device)
        
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr_actor, weight_decay=self.weight_decay)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=self.lr_critic, weight_decay=self.weight_decay)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=self.lr_critic, weight_decay=self.weight_decay)  # Оптимизатор для второго критика
        
    def get_action(self,state):
        state = torch.tensor(state, dtype=torch.float32).to(device)
        action = self.actor(state).detach().cpu().numpy() + self.noise.sample()
        return np.clip(a=action,a_min = -self.action_max, a_max = self.action_max)
    
    def soft_update(self, source, target, tau):
        """Softly update target network parameters."""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
            
    def update_target_networks(self):
        """Update the target networks."""
        self.soft_update(self.actor  , self.actor_target  , self.tau)
        self.soft_update(self.critic1, self.critic1_target, self.tau)
        self.soft_update(self.critic2, self.critic2_target, self.tau)
    
    def update(self):
        
        transitions, indices, weights = self.replay_buffer.sample(self.batch_size)
        if transitions is None:
            return
        
        states, actions, rewards, next_states, dones = zip(*transitions)
        
        states = torch.tensor(np.array(states), dtype=torch.float32).to(device)
        actions = torch.tensor(np.array(actions), dtype=torch.float32).to(device)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32).to(device)
        rewards = torch.tensor(np.array(rewards), dtype=torch.float32).to(device)
        dones = torch.tensor(np.array(dones), dtype=torch.float32).to(device)
        weights = torch.tensor(np.array(weights), dtype=torch.float32).to(device)
        
        target_q_values=[]
        
        # Update Critic
        next_q_values1 = self.critic1_target(next_states, actions)
        next_q_values2 = self.critic2_target(next_states, actions)
        for i in range(len(rewards)):
            target_q_values.append(rewards[i] + self.gamma * (1 - dones[i]) * torch.min(next_q_values1, next_q_values2)[i])  # Усреднение ошибок
        target_q_values = torch.tensor(target_q_values).to(device)
        
        # Calcualte error learn
        critic1_loss = (weights * nn.MSELoss(reduction='none')(self.critic1(states, actions), target_q_values.detach().unsqueeze(1))).mean()
        critic2_loss = (weights * nn.MSELoss(reduction='none')(self.critic2(states, actions), target_q_values.detach().unsqueeze(1))).mean()
        self.critic1_optimizer.zero_grad()
        self.critic2_optimizer.zero_grad()
        critic1_loss.backward()
        critic2_loss.backward()
        self.critic1_optimizer.step()
        self.critic2_optimizer.step()
        
        # Update Actor
        actors_1 = self.actor(states)
        actor_loss = -(weights * self.critic1(states, actors_1)).mean()  # Используем первый критик для обновления актера
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
        self.writer.add_scalar('Average Reward', torch.mean(rewards), self.global_step)
        self.global_step += 1
    
    def save_models(self):
        self.actor.save_checkpoint()
        self.actor_target.save_checkpoint()
        self.critic1.save_checkpoint()
        self.critic2.save_checkpoint()
        self.critic1_target.save_checkpoint()
        self.critic2_target.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.actor_target.load_checkpoint()
        self.critic1.load_checkpoint()
        self.critic2.load_checkpoint()
        self.critic1_target.load_checkpoint()
        self.critic1_target.load_checkpoint()
           
    def train(self, env, num_episodes=EPISODES, ep_steps=EP_STEPS):
        # Загрузка весов и моделей для продолжения обучения
        if os.path.exists('/chekpoints/actor*') :
        #     self.actor.load_state_dict(torch.load('actor_weights.pth'))
        #     self.critic1.load_state_dict(torch.load('critic1_weights.pth'))
        #     self.critic2.load_state_dict(torch.load('critic2_weights.pth'))
            self.load_models()
        
        start_time = time.time()
        episode_rewards = []
        
        for episode in range(num_episodes):
            state = env.reset_env()
            done = False
            episode_reward = 0
            env.set_number(episode)
            
            # for _ in range(ep_steps):
            while not done:
                action =self.get_action(state) 
                next_state, reward, done, _ = env.step(action)
                
                self.replay_buffer.push(state, action, reward, next_state, done)
                if len(self.replay_buffer) > self.batch_size:
                    self.update()
                state = next_state
                episode_reward += reward
            episode_rewards.append(episode_reward)
            avg_reward = np.mean(episode_rewards[-100:])
            
            print(f"Episode {episode+1}/{num_episodes}, Reward: {episode_reward:.2f}, Avg Reward: {avg_reward:.2f}")

            if episode % 5 == 0:
                # Save model
                # torch.save(self.actor.state_dict()  , 'actor_weights.pth')
                # torch.save(self.critic1.state_dict(), 'critic1_weights.pth')
                # torch.save(self.critic2.state_dict(), 'critic2_weights.pth')
                self.save_models()
            
        # Сохранение весов и моделей 
        # torch.save(self.actor.state_dict()  ,'actor_weights.pth')
        # torch.save(self.critic1.state_dict(),  'critic1_weights.pth')
        # torch.save(self.critic2.state_dict(),  'critic2_weights.pth')
        self.save_models()
        
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
        
        # self.actor.load_state_dict(torch.load( 'actor_weights.pth'))
        # self.critic1.load_state_dict(torch.load('critic1_weights.pth'))
        # self.critic2.load_state_dict(torch.load( 'critic2_weights.pth'))
        self.load_models()
       
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
