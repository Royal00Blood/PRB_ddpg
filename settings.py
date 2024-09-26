# Буфер-параметры
BUFFER_SIZE = 1000000 #от 10000 до 1000000
BATCH_SIZE  = 100 # 32 to 256
ALPHA = 0.6

# Гиперпараметры
GAMMA = 0.99   # 0.95 to 0.999
TAU   = 0.006 # 0.001 to 0.01

LR_A = 0.0006 # 0.0001 to 0.001
LR_C = 0.006 # 0.001 to 0.01

WEIGHT_DEC = 0.001 # 0.0001 0.01
REWARD = 500
# Параметры среды
S_SIZE  = 5
A_SIZE = 2

A_MAX = 0.5
S_MAX  = 10.0

#диапазон ограничений поля
AREA_WIN    = 0.1
AREA_DEFEAT = 7
S_G_TARG = AREA_WIN * 3

AREA_GENERATION = AREA_DEFEAT - 0.2
TIME = 0.1

# Параметры обучения
EPISODES = 1000
EP_STEPS = 500

#  Параметры проверки
TEST_EPISODES = 100
TEST_EP_STEPS = 200

# Параметры моделей
SEED = 200

# Кол нейронов в слоях
    #[ly1, ly2, ly3(v/w), ly4(v/w), ly5(v/w)] 
L_A  = [400, 300, 200, 100]
L_C1 = [300, 150, 100]
L_C2 = [150, 150, 100]

