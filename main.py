from EnvCustom import CustomEnv 
from Multi_Critic_DDPG import PRB_DDPG_Agent as Agent
from settings import  EPISODES, TEST_EP_STEPS, TEST_EPISODES
from Agent import Agent as Agent2 

env = CustomEnv()
agent = Agent()
agent.train(env)
agent.test(env,max_episodes=TEST_EPISODES,max_steps=TEST_EP_STEPS)
