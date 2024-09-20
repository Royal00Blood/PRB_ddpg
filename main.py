#from EnvESP32 import CustomEnv as env_real
from env_s.EnvCustom import CustomEnv as env_learn
from agents.Multi_Critic_DDPG import PRB_DDPG_Agent as Agent
from settings import  EPISODES, TEST_EP_STEPS, TEST_EPISODES

import torch._dynamo
torch._dynamo.config.suppress_errors = True
i, g, t, la, lc = 0,0,0,0,0
batch_s = 32
gamma = 0.9
tau = 0.001
lactor= 0.0001
lcritic= 0.001
env = env_learn()
for i in range(224):
    batch_s += i
    for g in range(99):
        gamma += g/1000 
        for t in range(10):
            tau += t/1000
            for la in range(10):
                lactor += la/10000
                for lc in range(10):
                    lcritic += lc/1000
                    agent = Agent(batch_size=batch_s, gamma=gamma, tau=tau, lr_actor=lactor, lr_critic=lcritic)
                    agent.train(env, num_episodes =250)
#agent.test(env,max_episodes=TEST_EPISODES,max_steps=TEST_EP_STEPS)
