# Буфер-параметры
BUFFER_SIZE = 100000 # от 10000 до 1000000
BATCH_SIZE  = 100     # 32 to 256
ALPHA = 0.65

# Гиперпараметры
GAMMA = 0.99   # 0.9 to 0.999
TAU   = 0.001  # 0.001 to 0.01

LR_A = 0.0001  # 0.0001 to 0.001
LR_C = 0.001   # 0.001 to 0.01

WEIGHT_DEC = 0.005 # 0.0001 0.01

# Параметры среды
S_SIZE  = 5
A_SIZE = 1

A_MAX = 0.5
S_MAX  = 10.0

#диапазон ограничений поля
AREA_WIN    = 0.1
AREA_DEFEAT = 1.8
S_G_TARG = AREA_WIN * 4

AREA_GENERATION = AREA_DEFEAT - S_G_TARG
TIME = 0.1

# Параметры обучения
EPISODES = 4000
EP_STEPS = 400

#  Параметры проверки
TEST_EPISODES = 100
TEST_EP_STEPS = 400

# Параметры моделей
SEED = 10
REWARD = EP_STEPS * 2

# Кол нейронов в слоях
    #[ly1, ly2, ly3(v/w), ly4(v/w), ly5(v/w)] 
L_A  = [512, 500, 450, 250]
L_C1 = [400, 300, 200]


DIR_CHEKPOINT = "C:/Users/Ivan/Documents/python_github/PRB_ddpg/chekpoints"
N_DIC = 1 / (EPISODES * EP_STEPS)