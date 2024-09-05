#from EnvESP32 import CustomEnv as env_real
from EnvCustom import CustomEnv as env_learn
from Multi_Critic_DDPG import PRB_DDPG_Agent as Agent
from settings import  EPISODES, TEST_EP_STEPS, TEST_EPISODES
from Agent import Agent as Agent2 
# from torchrl.modules import DdpgMlpActor
from settings import (STATE_SIZE, ACTION_SIZE, 
                      ACTION_, STATE_, AREA_DEFEAT, 
                      AREA_WIN, AREA_GENERATION, 
                      TIME, EP_STEPS )

env = env_learn()
agent = Agent()
agent.train(env)
#agent.test(env,max_episodes=TEST_EPISODES,max_steps=TEST_EP_STEPS)
