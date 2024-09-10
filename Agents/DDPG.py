import torch as th
import numpy as np 
from models.actors import Actor_1 as Actor
from models.critics import Critic1 as Critic
from buffers.experience_replay import replay_memory

from settings import (STATE_SIZE,ACTION_SIZE, ACTION_, BUFFER_SIZE,BATCH_SIZE,LR_ACTOR,LR_CRITIC,GAMMA,TAU)
device="cuda"
n_state = STATE_SIZE
n_action = ACTION_SIZE
max_action = ACTION_
memory_size = BUFFER_SIZE
lra = LR_ACTOR
lrc = LR_CRITIC
gamma = GAMMA
tau = TAU
batchsize=BATCH_SIZE

class DDPG():
    def __init__(self):
        self.actor=Actor(n_state,n_action,max_action).to(device)
        self.target_actor=Actor(n_state,n_action,max_action).to(device)
        self.critic=Critic(n_state,n_action).to(device)
        self.target_critic=Critic(n_state,n_action).to(device)
        self.memory=replay_memory(memory_size)
        self.Aoptimizer=th.optim.Adam(self.actor.parameters(),lr=lra)
        self.Coptimizer=th.optim.Adam(self.critic.parameters(),lr=lrc)

    def actor_learn(self,batch):
        b_s=th.FloatTensor(batch[:,0].tolist()).to(device)
        action=self.actor(b_s)
        #print(action)
        loss=-(self.critic(b_s,action).mean())
        self.Aoptimizer.zero_grad()
        loss.backward()
        self.Aoptimizer.step()
    
    def critic_learn(self,batch):
        b_s=th.FloatTensor(batch[:,0].tolist()).to(device)
        b_r=th.FloatTensor(batch[:,1].tolist()).to(device)
        b_a=th.FloatTensor(batch[:,2].tolist()).to(device)
        b_s_=th.FloatTensor(batch[:,3].tolist()).to(device)
        b_d=th.FloatTensor(batch[:,4].tolist()).to(device)

        next_action=self.target_actor(b_s_)
        #print(next_action)
        target_q=self.target_critic(b_s_,next_action)
        for i in range(b_d.shape[0]):
            if b_d[i]:
                target_q[i]=b_r[i]
            else:
                target_q[i]=b_r[i]+gamma*target_q[i]
        eval_q=self.critic(b_s,b_a)

        td_error=eval_q-target_q.detach()
        loss=(td_error**2).mean()
        self.Coptimizer.zero_grad()
        loss.backward()
        self.Coptimizer.step()

    def soft_update(self):
        for param,target_param in zip(self.actor.parameters(),self.target_actor.parameters()):
            target_param.data.copy_(tau*param.data+(1-tau)*target_param.data)
        for param,target_param in zip(self.critic.parameters(),self.target_critic.parameters()):
            target_param.data.copy_(tau*param.data+(1-tau)*target_param.data)
    
    def train(self,env):
        var=3
        for episode in range(2000):
            s=env.reset()
            total_reward=0
            Normal=th.distributions.normal.Normal(th.FloatTensor([0]),th.FloatTensor([var]))
            t=0
            while 1:
                noise=th.clamp(Normal.sample(),-max_action, max_action).to(device)
                a=self.actor(th.FloatTensor(s).to(device))+noise
                a=th.clamp(a,env.action_space.low[0], env.action_space.high[0]).to(device)
                
                s_,r,done,_=env.step(a.tolist())
                total_reward+=r
                transition=[s,[r],[a],s_,[done]]
                self.memory.store(transition)
                #print(done)
                if done:
                    break
                s=s_
                
                var*=0.9995
                batch=self.memory.sample(batchsize)
                self.critic_learn(batch)
                self.actor_learn(batch)

                self.soft_update()
                t+=1
            print("episode:"+format(episode)+",test score:"+format(total_reward)+',variance:',var)
