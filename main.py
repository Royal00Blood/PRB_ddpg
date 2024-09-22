#from EnvESP32 import CustomEnv as env_real
from env_s.EnvCustom import CustomEnv as env_learn
from Agents.Multi_Critic_DDPG import PRB_DDPG_Agent as Agent
from settings import  EPISODES, TEST_EP_STEPS, TEST_EPISODES

import torch._dynamo
torch._dynamo.config.suppress_errors = True
i, g, t, la, lc = 0,0,0,0,0
# batch_s = 32
gamma = 0.9
tau = 0.001
lactor= 0.0001
lcritic= 0.001
env = env_learn()
# for i in range(22):
#     batch_s += 10
for g in range(9):
    gamma = 0.9+g/100
    for t in range(10):
        tau = 0.001+t/1000
        for la in range(8):
            lactor = 0.0002+ la/10000
            for lc in range(8):
                lcritic = 0.002+ lc/1000
                env = env_learn()
                agent = Agent( gamma=gamma, tau=tau, lr_actor=lactor, lr_critic=lcritic)
                agent.train(env, num_episodes =200)
            
#agent.test(env,max_episodes=TEST_EPISODES,max_steps=TEST_EP_STEPS)
