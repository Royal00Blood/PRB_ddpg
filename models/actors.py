import torch
import os
import torch.nn as nn
import torch._dynamo
import torch.nn.functional as F
from settings import (A_MAX, S_SIZE, A_SIZE, SEED, L_A, DIR_CHEKPOINT,NUM_PARAMETERS,INIT)
torch._dynamo.config.suppress_errors = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
class Actor(nn.Module):
    """ Арихитектура модели Актера для сети DDPG принимает на вход:
        state_size  - размерность вектора состояния 
        action_size - размерность вектора действия
        seed        - параметр "скорости" обучения
        action_max  - максимальное значение действия по модулю
        layers      - вектор со значениями количества нейронов в каждом слое
    """
    def __init__(self, state_size, action_size, seed, action_max, 
                 layers, dir_chekpoint, name, function_a):
        super(Actor, self).__init__()
        self.tanh = nn.Tanh()
        self.function_a = function_a
        self.chekpoint = os.path.join(dir_chekpoint, name)
        self.seed = torch.manual_seed(seed)
        self.action_max = action_max
        
        # Инициализация слоев и нормализации
        self.layers = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        # Создание слоев
        for i in range(len(layers) - 1):
            self.layers.append(nn.Linear(layers[i], layers[i + 1]))
            self.batch_norms.append(nn.LayerNorm(layers[i + 1]))

        # Выходной слой
        self.output_layer = nn.Linear(layers[-1], action_size)

        # Инициализация весов
        self.reset_weights()
    
    def reset_weights(self):
        
        # Инициализация весов для скрытых слоев
        for layer in self.layers:
            if self.function_a == nn.LeakyReLU:
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='leaky_relu')
            elif self.function_a == nn.ReLU:
                nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
            nn.init.constant_(layer.bias, 0.1)
            # Проверяем, является ли слой PReLU и если да, то инициализируем альфа
            if isinstance(layer, nn.PReLU):
                nn.init.constant_(layer.weight, INIT)
            
         # Инициализация выходного слоя
        nn.init.uniform_(self.output_layer.weight.data, -0.1, 0.1)
        
    def forward(self, state):
        x = state
        # Применение скрытых слоев и нормализации
        for layer, batch_norm in zip(self.layers, self.batch_norms):
            x = self.function_a(batch_norm(layer(x)))
        action = self.tanh(self.output_layer(x)) * self.action_max 
        return action

    def save_checkpoint(self):
        torch.save(self.state_dict(), self.chekpoint)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.chekpoint))