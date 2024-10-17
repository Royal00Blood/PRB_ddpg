#from EnvESP32 import CustomEnv as env_real
from env_s.EnvCustom import CustomEnv as env_learn
from Agents.Multi_Critic_DDPG import PRB_DDPG_Agent as Agent
from settings import  EPISODES, TEST_EP_STEPS, TEST_EPISODES

import torch._dynamo
torch._dynamo.config.suppress_errors = True

env = env_learn()
agent = Agent()
agent.train(env)               
#agent.test(env,max_episodes=TEST_EPISODES,max_steps=TEST_EP_STEPS)
