# Буфер-параметры
BUFFER_SIZE = 1000000 #от 10000 до 1000000
BATCH_SIZE  = 70 # 32 to 256
ALPHA = 0.6

# Гиперпараметры
GAMMA = 0.99   # 0.95 to 0.999
TAU   = 0.007 # 0.001 to 0.01

LR_ACTOR  = 0.0009 # 0.0001 to 0.001
LR_CRITIC = 0.009 # 0.001 to 0.01

WEIGHT_DEC = 0.003 # 0.0001 0.01
REWARD = 1000
# Параметры среды
STATE_SIZE  = 5
ACTION_SIZE = 1

ACTION_ = 0.5
STATE_  = 10.0

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
LAYER_A1 = [300, 100, 50]
LAYER_A2 = [400, 250, 300]
LAYER_C1 = [250, 100]
LAYER_C2 = [200, 150]

