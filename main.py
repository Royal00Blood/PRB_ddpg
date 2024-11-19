from env_s.EnvCustom import CustomEnv as env_learn
#from env_s.EnvESP32 import CustomEnv as env_esp
from Agents.Multi_Critic_DDPG import PRB_DDPG_Agent as Agent
from settings import  EPISODES, TEST_EP_STEPS, TEST_EPISODES, S_G_TARG, AREA_GENERATION
import random

import torch._dynamo
torch._dynamo.config.suppress_errors = True

#target_point = [random.uniform(S_G_TARG, AREA_GENERATION), random.uniform(S_G_TARG, AREA_GENERATION)] 

env = env_learn()
#envesp = env_esp(target_point)
agent = Agent()
agent.train(env)
agent.test(env,max_episodes=100, max_steps=TEST_EP_STEPS)
#agent.test(envesp,max_episodes=1, max_steps=TEST_EP_STEPS)

