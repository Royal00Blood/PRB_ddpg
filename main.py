#from EnvESP32 import CustomEnv as env_real
from env_s.EnvCustom import CustomEnv as env_learn
from Agents.Multi_Critic_DDPG import PRB_DDPG_Agent as Agent
from settings import  EPISODES, TEST_EP_STEPS, TEST_EPISODES

import torch._dynamo
torch._dynamo.config.suppress_errors = True

env = env_learn()
agent = Agent()
agent.train(env) 
i = 0
a = 0.0001
c = 0.001
for i in range(10):
    a = 0.0001 + 0.0001 * i  
    j = 0
    for j in range(10):
        c =  0.001 + 0.001 * j
        agent = Agent(lr_actor=a,lr_critic=c)
        agent.train(env, num_episodes=200)  
                    
#agent.test(env,max_episodes=TEST_EPISODES,max_steps=TEST_EP_STEPS)
