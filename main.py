#from EnvESP32 import CustomEnv as env_real
from env_s.EnvCustom import CustomEnv as env_learn
from agents.Multi_Critic_DDPG import PRB_DDPG_Agent as Agent
from agents.DDPG import DDPG as Agent
from settings import  EPISODES, TEST_EP_STEPS, TEST_EPISODES


env = env_learn()
agent = Agent()
agent.train(env)
#agent.test(env,max_episodes=TEST_EPISODES,max_steps=TEST_EP_STEPS)
