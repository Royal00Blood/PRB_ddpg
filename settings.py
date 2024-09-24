# Буфер-параметры
BUFFER_SIZE = 1_000_000 #от 10000 до 1000000
BATCH_SIZE  = 160  # 32 to 256
ALPHA = 0.6

# Гиперпараметры
GAMMA = 0.95   # 0.95 to 0.999
TAU   = 0.005 # 0.001 to 0.01

LR_ACTOR  = 0.0007 # 0.0001 to 0.001
LR_CRITIC = 0.007 # 0.001 to 0.01

WEIGHT_DEC = 0.001 # 
REWARD = 500
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
EPISODES = 500
EP_STEPS = 500

#  Параметры проверки
TEST_EPISODES = 100
TEST_EP_STEPS = 200

# Параметры моделей
SEED = 200

# Кол нейронов в слоях
         #[ly1, ly2, ly3(v/w), ly4(v/w), ly5(v/w)] 
LAYER_A1 = [300, 150, 200]
LAYER_A2 = [400, 250, 300]
LAYER_C1 = [250,200]
LAYER_C2 = [300,150]

