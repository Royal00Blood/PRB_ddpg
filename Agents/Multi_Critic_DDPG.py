import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from models.actors import Actor
from models.critics import Critic
import os
from settings import (S_SIZE, A_SIZE, LR_A, LR_C, L_C1,
                      BATCH_SIZE,GAMMA,BUFFER_SIZE,
                      ALPHA, TAU, EPISODES,
                      EP_STEPS, TEST_EP_STEPS,
                      TEST_EPISODES, A_MAX, WEIGHT_DEC,N_DIC)
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
                 weight_decay = WEIGHT_DEC, action_max=A_MAX, n_dic=N_DIC):
        
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
        self.layers_critic = L_C1
        self.noise = Noise(self.action_size)
        self.n_treshold =1
        self.n_dic = n_dic
        
        self.actor = torch.compile(Actor(self.state_size, self.action_size)).to(device)
        self.critic = torch.compile(Critic(self.state_size, self.action_size,layers=self.layers_critic, name="critic")).to(device)
        self.actor_target = torch.compile(Actor(state_size, action_size,name="actor_target")).to(device)
        self.critic_target = torch.compile(Critic(self.state_size, self.action_size,layers=self.layers_critic,name="critic_target")).to(device)
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr_actor, weight_decay=self.weight_decay)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.lr_critic, weight_decay=self.weight_decay)
        
    def get_action(self,state):
        state = torch.FloatTensor(state).to(device)
        action = self.actor(state).detach().cpu().numpy() 
        action_n = action + self.noise.sample()*self.n_treshold
        action = np.clip(a=action_n,a_min = -self.action_max, a_max = self.action_max)
        return action
      
    def update_target_networks(self):
        """Update the target networks."""
        self.soft_update(self.actor  , self.actor_target, self.tau)
        self.soft_update(self.critic, self.critic_target, self.tau)
        
    def soft_update(self, source, target, tau):
        """Softly update target network parameters."""
        for target_param, param in zip(target.parameters(), source.parameters()):
            target_param.data.copy_(target_param.data * (1.0 - tau) + param.data * tau)
            
    def update(self):
        transitions, indices, weights = self.replay_buffer.sample(self.batch_size)
        if transitions is None:
            return
        
        states, actions, rewards, next_states, dones = zip(*transitions)
        states = torch.FloatTensor(states).to(device)
        actions = torch.FloatTensor(actions).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        rewards = rewards.reshape(self.batch_size,1)
        dones = torch.FloatTensor(dones).to(device)
        dones = dones.reshape(self.batch_size,1)
        weights =  torch.FloatTensor(weights).to(device)
        
        # Update Critic
        next_action = self.actor_target(next_states)
        next_states_action = torch.cat((next_states,next_action), dim=1)
        next_q_values = self.critic_target(next_states_action)
        targets = rewards + (1-dones) * self.gamma * next_q_values
        
        # Calcualte error learn
        states_actions = torch.cat((states, actions),dim=1).to(device)
        critic_loss = torch.mean(weights *(targets.detach() - self.critic(states_actions))**2).to(device)
        critic_loss.backward()
        self.critic_optimizer.step()
        self.critic_optimizer.zero_grad()
        
        # Update Actor
        actors_1 = self.actor(states).to(device)
        states_actors_1 = torch.cat((states, actors_1),dim=1).to(device)
        actor_loss = -torch.mean(weights * self.critic(states_actors_1)).to(device)  # Используем первый критик для обновления актера
        actor_loss.backward()
        self.actor_optimizer.step()
        self.actor_optimizer.zero_grad()
        ##
        self.update_target_networks()
        # Update Priorities
        new_priorities = (critic_loss.detach().cpu().numpy()  + 1e-5) / 2
        self.replay_buffer.update_priority(indices, new_priorities)
        
        self.writer.add_scalar('Actor Loss'    , actor_loss.item()  , self.global_step)
        self.writer.add_scalar('Critic Loss'   , critic_loss.item() , self.global_step)
        self.writer.add_scalar('Average Reward', torch.mean(rewards), self.global_step)
        self.global_step += 1
    
    def save_models(self):
        self.actor.save_checkpoint()
        self.actor_target.save_checkpoint()
        self.critic.save_checkpoint()
        self.critic_target.save_checkpoint()

    def load_models(self):
        self.actor.load_checkpoint()
        self.actor_target.load_checkpoint()
        self.critic.load_checkpoint()
        self.critic_target.load_checkpoint()
           
    def train(self, env, num_episodes=EPISODES, ep_steps=EP_STEPS):
        # Загрузка весов и моделей для продолжения обучения
        if os.path.exists('/chekpoints'):
            if os.path.exists('actor_weights.pth'):
                self.actor.load_state_dict(torch.load('actor_weights.pth'))
                self.critic1.load_state_dict(torch.load('critic_weights.pth'))
            self.load_models()
        
        start_time = time.time()
        episode_rewards = []
        
        for episode in range(num_episodes):
            state = env.reset_env()
            done = False
            episode_reward = 0
            env.set_number(episode)
            
            
            for _ in range(ep_steps):# попробовать обучить при while
            # while not done:
                if done:
                    break
                action =self.get_action(state) 
                next_state, reward, done, _ = env.step(action)
                
                self.replay_buffer.push(state, action, reward, next_state, done)
                
                if len(self.replay_buffer) > self.batch_size:# Добавить обновление в случае кратности шага после проверки а размер буфера
                    self.update()
                    
                state = next_state
                episode_reward += reward
            episode_rewards.append(episode_reward)
            avg_reward = np.mean(episode_rewards[-100:])
            
            print(f"Episode {episode+1}/{num_episodes}, Reward: {episode_reward:.2f}, Avg Reward: {avg_reward:.2f}, Target point {state[2]:.2f} , {state[3]:.2f}")

            if episode % 5 == 0:
                # Save model
                torch.save(self.actor.state_dict()  , 'actor_weights.pth')
                torch.save(self.critic.state_dict(), 'critic_weights.pth')
                self.save_models()
                
        if self.n_treshold>0:
            self.n_treshold = max(0,self.n_treshold-self.n_dic)    
            
        # Сохранение весов и моделей 
        torch.save(self.actor.state_dict()  ,'actor_weights.pth')
        torch.save(self.critic.state_dict(),  'critic_weights.pth')
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
        
        self.actor.load_state_dict(torch.load( 'actor_weights.pth'))
        self.critic.load_state_dict(torch.load('critic_weights.pth'))
        self.load_models()
       
        rewards = []
       
        for episode in range(max_episodes):
            state = env.reset_env()
            total_reward = 0
            done = False
            for st in range(max_steps):
            #while(not done):
                # Выбираем действие на основе текущего состояния
                action = self.get_action(state)
                # Применяем действие в среде и получаем новое состояние, награду и флаг завершения
                next_state, reward, done, _ = env.step(action)
                total_reward += reward
                if done:
                    break
                state = next_state
                
            rewards.append(total_reward)
            print(f"Episode {episode+1}, Reward: {total_reward:.2f}, Target point {state[1:3]:.2f}")
        
        print(f"Average reward: {np.mean(rewards):.2f}")
