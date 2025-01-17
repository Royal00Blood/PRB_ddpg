import torch.nn as nn

# Буфер-параметры
BUFFER_SIZE = 100000 # от 10000 до 1000000
BATCH_SIZE  = 150     # 32 to 256 #100
ALPHA = 0.6 #65

# Гиперпараметры
GAMMA = 0.99   # 0.9 to 0.999
TAU   = 0.001  # 0.001 to 0.01

LR_A = 0.0001  # 0.0001 to 0.001
LR_C_1 = 0.001   # 0.001 to 0.01
LR_C_2 = 0.001   # 0.001 to 0.01

WEIGHT_DEC = 0.005 # 0.0001 0.01

# Для  PRelu:
NUM_PARAMETERS=1
INIT = 0.25 # 

# Параметры среды
S_SIZE  = 5
A_SIZE = 2

A_MAX = 0.5
S_MAX  = 10.0

#диапазон ограничений поля
AREA_WIN    = 0.05
AREA_DEFEAT = 1.8
S_G_TARG = 0.2
ANGL_WIN = 0.02

AREA_GENERATION = AREA_DEFEAT - S_G_TARG
TIME = 0.1

# Параметры обучения
EPISODES = 10000
EP_STEPS = 400

#  Параметры проверки
TEST_EPISODES = 100
TEST_EP_STEPS = 400

# Параметры моделей
SEED = 10
REWARD = EP_STEPS * 2


# Кол нейронов в слоях
    #[ly1, ly2, ly3(v/w), ly4(v/w), ly5(v/w)] 
L_A  = [200,350,500]
L_C1 = [400,350,200] 
L_C2 = [500,400,100]


DIR_CHEKPOINT = "chekpoints"
NAME_CHEKPOINT_A  = "actor_chekpoint"
NAME_CHEKPOINT_C1 = "critic_chekpoint_1"
NAME_CHEKPOINT_C2 = "critic_chekpoint_2"
T_NAME_CHEKPOINT_A  = "T_actor_chekpoint"
T_NAME_CHEKPOINT_C1 = "T_critic_chekpoint_1"
T_NAME_CHEKPOINT_C2 = "T_critic_chekpoint_2"
N_DIC = 1 / (EPISODES * EP_STEPS)