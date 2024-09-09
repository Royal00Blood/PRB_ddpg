import torch
import torch.optim as optim
import torch.nn as nn
import numpy as np
import os
import time
from torch.utils.tensorboard import SummaryWriter
from torchrl.data import LazyMemmapStorage, TensorDictReplayBuffer
from torchrl.data.replay_buffers.samplers import PrioritizedSampler

from model import Actor, Critic1, Critic2
from settings import (STATE_SIZE, ACTION_SIZE, LR_ACTOR,
                      LR_CRITIC,BATCH_SIZE,GAMMA,BUFFER_SIZE,
                      ALPHA,TAU,EPISODES,EP_STEPS,TEST_EP_STEPS,
                      TEST_EPISODES)
import torch
from tensordict import TensorDict


class Agent:
    def __init__(self, state_size=STATE_SIZE, action_size=ACTION_SIZE, lr_actor=LR_ACTOR,
                 lr_critic=LR_CRITIC, gamma=GAMMA, tau=TAU, buffer_size=BUFFER_SIZE,
                 batch_size=BATCH_SIZE, alpha=ALPHA):
        
        self.state_size = state_size
        self.action_size = action_size
        self.lr_actor = lr_actor
        self.lr_critic = lr_critic
        self.gamma = gamma
        self.tau = tau
        self.buffer_size = buffer_size
        self.batch_size = batch_size
        self.alpha = alpha

        # Define Replay Buffer
        self.storage = LazyMemmapStorage(self.buffer_size)
        self.sampler = PrioritizedSampler(self.buffer_size, alpha, beta=1.0)
        self.replay_buffer = TensorDictReplayBuffer(storage=self.storage, sampler=self.sampler)

        # Define Networks
        self.actor = torch.compile(Actor(self.state_size, self.action_size))
        self.critic1 = torch.compile(Critic1(self.state_size, self.action_size))
        self.critic2 = torch.compile(Critic2(self.state_size, self.action_size))
        self.actor_target = torch.compile(Actor(self.state_size, self.action_size))
        self.critic1_target = torch.compile(Critic1(self.state_size, self.action_size))
        self.critic2_target = torch.compile(Critic2(self.state_size, self.action_size))

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.lr_actor)
        self.critic1_optimizer = optim.Adam(self.critic1.parameters(), lr=self.lr_critic)
        self.critic2_optimizer = optim.Adam(self.critic2.parameters(), lr=self.lr_critic)

        # Summary Writer
        self.writer = SummaryWriter()
        self.global_step = 0

    def update(self):
        transitions, indices, weights = self.replay_buffer.sample(self.batch_size)
        if transitions is None:
            return
        states, actions, rewards, next_states, dones = zip(*transitions)
        
        states = torch.tensor(states, dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(next_states, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)
        weights = torch.tensor(weights, dtype=torch.float32)

        # Update Critic
        next_actions = self.actor_target(next_states)
        next_q_values1 = self.critic1_target(next_states, next_actions)
        next_q_values2 = self.critic2_target(next_states, next_actions)
        target_q_values = rewards + self.gamma * (1 - dones) * torch.min(next_q_values1, next_q_values2)

        critic1_loss = (weights * nn.MSELoss(reduction='none')(self.critic1(states, actions), target_q_values.detach())).mean()
        critic2_loss = (weights * nn.MSELoss(reduction='none')(self.critic2(states, actions), target_q_values.detach())).mean()

        self.critic1_optimizer.zero_grad()
        critic1_loss.backward()
        self.critic1_optimizer.step()

        self.critic2_optimizer.zero_grad()
        critic2_loss.backward()
        self.critic2_optimizer.step()

        # Update Actor
        actor_loss = -(weights * self.critic1(states, self.actor(states))).mean()
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        
        # Update Priorities
        new_priorities = (critic1_loss.detach().cpu().numpy() + 1e-5)
        self.replay_buffer.update_priority(indices, new_priorities)

        # Update Target Networks
        for target_param, param in zip(self.actor_target.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
        for target_param, param in zip(self.critic1_target.parameters(), self.critic1.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
        for target_param, param in zip(self.critic2_target.parameters(), self.critic2.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
        
        # Log metrics
        self.writer.add_scalar('Actor Loss', actor_loss.item(), self.global_step)
        self.writer.add_scalar('Critic1 Loss', critic1_loss.item(), self.global_step)
        self.writer.add_scalar('Critic2 Loss', critic2_loss.item(), self.global_step)
        self.global_step += 1

    def train(self, env):
        if os.path.exists( 'actor_weights.pth') and os.path.exists( 'critic1_weights.pth') and os.path.exists( 'critic2_weights.pth'):
            self.actor.load_state_dict(torch.load('actor_weights.pth'))
            self.critic1.load_state_dict(torch.load('critic1_weights.pth'))
            self.critic2.load_state_dict(torch.load('critic2_weights.pth'))
        
        start_time = time.time()
        episode_rewards = []
        
        for episode in range(EPISODES):
            state = env.reset_env()
            done = False
            episode_reward = 0
            env.set_number(episode)
            
            for _ in range(EP_STEPS):
                action = self.actor(torch.tensor(state, dtype=torch.float32)).detach().numpy()
                next_state, reward, done, _ = env.step(action)
                
                # Create a TensorDict
                td = TensorDict({
                "obs": torch.tensor(state, dtype=torch.float32).unsqueeze(0),
                "next": {"obs": torch.tensor(next_state, dtype=torch.float32).unsqueeze(0)},
                "td_error": torch.tensor([0.0])  # Начальное значение td_error
            }, [1])  # Размерность батча равна 1
                
                # Extend the replay buffer
                self.replay_buffer.extend(td)
            
                if len(self.replay_buffer) > self.batch_size:
                    self.update()

                state = next_state
                episode_reward += reward
                if done:
                    break
            episode_rewards.append(episode_reward)
            avg_reward = np.mean(episode_rewards[-100:])
            
            print(f"Episode {episode+1}/{EPISODES}, Reward: {episode_reward:.2f}, Avg Reward: {avg_reward:.2f}")
            if episode % 10 == 0:
                torch.save(self.actor.state_dict(), os.path.join(save_dir, 'actor_weights.pth'))
                torch.save(self.critic1.state_dict(), os.path.join(save_dir, 'critic1_weights.pth'))
                torch.save(self.critic2.state_dict(), os.path.join(save_dir, 'critic2_weights.pth'))
            
        torch.save(self.actor.state_dict(), os.path.join(save_dir, 'actor_weights.pth'))
        torch.save(self.critic1.state_dict(), os.path.join(save_dir, 'critic1_weights.pth'))
        torch.save(self.critic2.state_dict(), os.path.join(save_dir, 'critic2_weights.pth'))
        
        self.writer.close()
        end_time = time.time()
        print(f"Training time: {end_time - start_time:.2f} seconds")
        return episode_rewards
