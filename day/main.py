#from EnvESP32 import CustomEnv as env_real
from ENV.EnvCustom import CustomEnv as env_learn
from Agents.Multi_Critic_DDPG import PRB_DDPG_Agent as Agent
from settings import  (EPISODES, TEST_EP_STEPS, TEST_EPISODES)

env = env_learn()
agent = Agent()
agent.train(env)
agent.test(env)
