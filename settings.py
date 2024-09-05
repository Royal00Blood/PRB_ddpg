# Буфер-параметры
BUFFER_SIZE = 100_000
BATCH_SIZE  = 256
ALPHA = 0.6

# Гиперпараметры
GAMMA = 0.999# 99
TAU   = 0.01
NOISE = 0.1

LR_ACTOR  = 0.0001
LR_CRITIC = 0.001

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