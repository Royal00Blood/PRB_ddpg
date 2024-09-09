# Буфер-параметры
BUFFER_SIZE = 100_000
BATCH_SIZE  = 250      # 32 to 256
ALPHA = 0.6

# Гиперпараметры
GAMMA = 0.99   # 0.9 to 0.999
TAU   = 0.005  # 0.001 to 0.01
NOISE = 0.1    # Гаусов шум 

LR_ACTOR  = 0.0006 # 0.0001 to 0.001
LR_CRITIC = 0.003 # 0.001 to 0.01

REWARD = 500
# Параметры среды
STATE_SIZE  = 7
ACTION_SIZE = 2

ACTION_ = 0.5
STATE_  = 10.0

#диапазон ограничений поля
AREA_WIN    = 0.1
AREA_DEFEAT = 1.5
S_G_TARG = AREA_WIN*3

AREA_GENERATION = AREA_DEFEAT-0.2
TIME = 0.1

# Параметры обучения
EPISODES = 6000
EP_STEPS = 500

#  Параметры проверки
TEST_EPISODES = 100
TEST_EP_STEPS = 200

# Параметры моделей
SEED = 200

# Кол нейронов в слоях
         #[ly1, ly2, ly3(v/w), ly4(v/w), ly5(v/w)] 
LAYER_A = [200, 250, 300, 400, 512]
LAYER_C1 = [200, 400]
LAYER_C2 = [250, 500]
