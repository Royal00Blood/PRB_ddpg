# Буфер-параметры
BUFFER_SIZE = 100_000
BATCH_SIZE  = 256 # 32 to 256
ALPHA = 0.6

# Гиперпараметры
GAMMA = 0.992# 0.9 to 0.999
TAU   = 0.001 # 0.001 to 0.01
NOISE = 0.1 # Гаусов шум 

LR_ACTOR  = 0.0001 # 0.0001 to 0.001
LR_CRITIC = 0.001 # 0.001 to 0.01

# Параметры среды
STATE_SIZE  = 7
ACTION_SIZE = 2

ACTION_ = 0.5
STATE_  = 10.0

AREA_WIN    = 0.1
AREA_DEFEAT = 1.5

AREA_GENERATION = AREA_DEFEAT-0.2
TIME = 0.1

# Параметры обучения
EPISODES = 6000
EP_STEPS = 2000

#  Параметры проверки
TEST_EPISODES = 100
TEST_EP_STEPS = 200

# Параметры моделей
SEED = 200
# нейроны в слоях актера
A_FC1 = 200
A_FC2 = 350
A_FC3 = 400
# нейроны в слоях критика 1
C_FC11 = 200
C_FC12 = 350
C_FC13 = 400
# нейроны в слоях критика 2 
C_FC21 = 200
C_FC22 = 350
C_FC23 = 400