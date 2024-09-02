import torch
import torch.nn as nn
import torch.nn.functional as F
import torch._dynamo
from settings import ACTION_
#torch._dynamo.config.suppress_errors = True

class Actor(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Actor, self).__init__()
        
        # Определение слоев сети
        self.fc1 = nn.Linear(input_dim, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, output_dim)
        self.action_range = [-ACTION_, ACTION_] #action range (min, max)
        # Инициализация параметров сети
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc3(x))  # Используем tanh для ограничения значений
        x = x * (self.action_range[1] - self.action_range[0]) / 2 + (self.action_range[1] + self.action_range[0]) / 2
        return x
    
class Critic(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(Critic, self).__init__()
        # Q-value network
        self.fc1 = nn.Linear(state_dim + action_dim, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, 1)
        
    def forward(self, state, action):
        # Combine state and action
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x


# class Actor(nn.Module):
#     def __init__(self, state_size, action_size):
#         super(Actor, self).__init__()
#         self.fc1 = nn.Linear(state_size, 100)
#         self.dropout = nn.Dropout(0.2)
#         self.fc2 = nn.Linear(1000, 150)
        
#         self.fc3 = nn.Linear(150, 200)
#         self.fc4 = nn.Linear(200, 250)
#         self.dropout3 = nn.Dropout(p=0.4)
        
#         self.fc5v = nn.Linear(250, 200)
#         self.fc5w = nn.Linear(250, 200)
#         self.bn5 = nn.LayerNorm(200)
        
#         self.fc6v = nn.Linear(200, 1)
#         self.fc6w = nn.Linear(200, 1)
        
#     def forward(self, state):
    
#         x = torch.relu(self.fc1(state))
#         #x = self.dropout(x)
#         x = torch.relu(self.fc2(x))
#         x = torch.relu(self.fc3(x))
#         x = torch.relu(self.fc4(x))
        
#         v = torch.relu(self.fc5v(x))
#         #v = self.bn5(v)
#         w = torch.relu(self.fc5w(x))
#         #w = self.bn5(w)
        
#         v = torch.tanh(self.fc6v(v))
#         w = torch.tanh(self.fc6w(w))
#         return torch.cat([v*ACTION_, w*ACTION_], dim=-1)
    
# class Critic(nn.Module):
#     def __init__(self, state_size, action_size):
#         super(Critic, self).__init__()
#         self.fc1 = nn.Linear(state_size + action_size, 200)
#         self.bn1 = nn.BatchNorm1d(200)
        
#         self.fc2 = nn.Linear(200, 250)
#         self.bn2 = nn.BatchNorm1d(250)
        
#         self.fc3 = nn.Linear(250, 1)
        
#     def forward(self, state, action):
#         x = torch.cat([state, action], dim=1)
#         x = torch.relu(self.bn1(self.fc1(x)))
#         x = torch.relu(self.bn2(self.fc2(x)))
#         return self.fc3(x)
