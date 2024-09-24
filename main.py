#from EnvESP32 import CustomEnv as env_real
from env_s.EnvCustom import CustomEnv as env_learn
from Agents.Multi_Critic_DDPG import PRB_DDPG_Agent as Agent
from settings import  EPISODES, TEST_EP_STEPS, TEST_EPISODES

import torch._dynamo
torch._dynamo.config.suppress_errors = True

t, la, lc = 0, 0, 0 

gamma = 0.99
t, la, lc = 0, 0, 0 
tau = 0.005
for la in range(10):
    lactor = 0.0001 + la * 0.0001
    lc = 0 
    for lc in range(10):
        lcritic = 0.001+ lc * 0.001
        env = env_learn()
        agent = Agent( gamma=gamma, tau=tau, lr_actor=lactor, lr_critic=lcritic)
        agent.train(env, num_episodes=50)
                  
#agent.test(env,max_episodes=TEST_EPISODES,max_steps=TEST_EP_STEPS)
