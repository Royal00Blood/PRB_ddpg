

# Гиперпараметры
BUFFER_SIZE = 1000
BATCH_SIZE  = 64

GAMMA = 0.99
TAU   = 0.001

LR_ACTOR  = 0.001
LR_CRITIC = 0.0001

# Параметры среды
STATE_SIZE  = 8
ACTION_SIZE = 2

ACTION_ = 0.5
STATE_  = 10.0

AREA_WIN    = 0.1
AREA_DEFEAT = 1.7

AREA_GENERATION = AREA_DEFEAT-0.2
TIME = 0.1

# Параметры обучения
EPISODES = 10000
EP_STEPS = 200

TEST_EPISODES = 100
TEST_EP_STEPS = 200

ALPHA = 0.6