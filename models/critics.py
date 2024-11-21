import torch
import os
import torch.nn as nn
import torch._dynamo
import torch.nn.functional as F
from settings import (S_SIZE, SEED, A_SIZE, DIR_CHEKPOINT,NUM_PARAMETERS,INIT)
import numpy as np
torch._dynamo.config.suppress_errors = False
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class Critic(nn.Module):
    def __init__(self, state_size, action_size, seed, layers, dir_chekpoint, name, init, activ_f):
        super(Critic, self).__init__()
        
        self.chekpoint = os.path.join(dir_chekpoint, name)
        self.seed = torch.manual_seed(seed)
        self.activ_f = activ_f
        # Список, который будет содержать все слои
        self.layers = nn.ModuleList()  
        self.batch_norms = nn.ModuleList()
        
        # Создание первого слоя
        self.layers.append(nn.Linear(state_size + action_size, layers[0]))
        self.batch_norms.append(nn.LayerNorm(layers[0]))
        
        # Создание скрытых слоев
        for i in range(1, len(layers)):
            self.layers.append(nn.Linear(layers[i - 1], layers[i]))
            self.batch_norms.append(nn.LayerNorm(layers[i]))
        
        # Финальный слой
        self.q = nn.Linear(layers[-1], 1)

        self.reset_parameters()
        
    def reset_parameters(self):
        for layer in self.layers:
            nn.init.kaiming_normal_(layer.weight, mode='fan_out', nonlinearity='relu')
            # Инициализируем смещения
            nn.init.constant_(layer.bias, 0.1)
            # # Проверяем, является ли слой PReLU и если да, то инициализируем альфа
            if isinstance(layer, nn.PReLU):
                nn.init.constant_(layer.weight, INIT)
                
    def forward(self, x):
        # Прямое распространение
        for i in range(len(self.layers)):
            x = self.layers[i](x)
            x = self.batch_norms[i](x)
            x = self.activ_f(x)  # Применение активации (например, ReLU)
        return self.q(x) 
    
    def save_checkpoint(self):
        torch.save(self.state_dict(), self.chekpoint)

    def load_checkpoint(self):
        self.load_state_dict(torch.load(self.chekpoint))    


