import torch
import torch.nn as nn
# import torch._dynamo
import torch.nn.functional as F
from settings import (ACTION_, STATE_SIZE, SEED,
                      ACTION_SIZE,LAYER_A)
# torch._dynamo.config.suppress_errors = True
    
class Actor_1(nn.Module):
    def __init__(self, 
                 state_size=STATE_SIZE, 
                 action_size=ACTION_SIZE, 
                 seed=SEED,
                 layers=LAYER_A ):
        super(Actor_1, self).__init__()
        
        self.seed = torch.manual_seed(seed)
        self.dropout = nn.Dropout(0.5)
        
        self.layer_1 = nn.Linear(state_size, layers[0])
        nn.init.kaiming_normal_(self.layer_1.weight, mode='fan_in',nonlinearity='relu')
        self.batch_norm_1 = nn.BatchNorm1d(layers[0])
        
        self.layer_2 = nn.Linear(layers[0],layers[1])
        nn.init.orthogonal_(self.layer_2.weight, gain=nn.init.calculate_gain('relu'))
        
        self.batch_norm_2 = nn.BatchNorm1d(layers[1])
        
        self.layer_3 = nn.Linear(layers[1],layers[2])
        nn.init.orthogonal_(self.layer_3.weight, gain=nn.init.calculate_gain('relu'))
        
        self.batch_norm_3 = nn.BatchNorm1d(layers[2])
        
        self.layer_4 = nn.Linear(layers[2],layers[3])
        nn.init.orthogonal_(self.layer_4.weight, gain=nn.init.calculate_gain('relu'))
        self.batch_norm_4 = nn.BatchNorm1d(layers[3])
        
        self.layer_5 = nn.Linear(layers[3],action_size)
        self.batch_norm_5 = nn.BatchNorm1d(layers[3])
        
        
    def forward(self, state):
        x = self.layer_1(state)
        #x = self.batch_norm_1(x)
        x = self.dropout(x)
        x = F.relu(x)
        
        x = self.layer_2(x)
        #x = self.batch_norm_2(x)
        x = self.dropout(x)
        x = F.relu(x)
        
        x = self.layer_3(x)
        #x = self.batch_norm_3(x)
        x = self.dropout(x)
        x = F.relu(x)
        
        x = self.layer_4(x)
        #x = self.batch_norm_4(x)
        x = self.dropout(x)
        x = F.relu(x)
        
        action = self.layer_5(x)
        #action = self.batch_norm_5(action)
        action = torch.tanh(action) * ACTION_
        
        return action


class Actor_2(nn.Module):
    def __init__(self, 
                 state_size = STATE_SIZE, 
                 action_size=ACTION_SIZE, 
                 seed=SEED,
                 layers=LAYER_A ):
        super(Actor_2, self).__init__()
        
        self.seed = torch.manual_seed(seed)
        
        self.layer_1 = nn.Linear(state_size, layers[0])
        torch.nn.init.kaiming_normal_(self.layer_1,mode='fan_in',nonlinearity='relu')
        self.batch_norm_1 = nn.BatchNorm1d(layers[0])
        
        self.layer_2 = nn.Linear(layers[0],layers[1])
        torch.nn.init.xavier_uniform_(self.layer_2.weight)
        self.dropout_2 = nn.Dropout(0.5)
        self.batch_norm_2 = nn.BatchNorm1d(layers[1])
        
        self.layer_3 = nn.Linear(layers[1],layers[2])
        torch.nn.init.xavier_uniform_(self.layer_3.weight)
        self.dropout = nn.Dropout(0.5)
        self.batch_norm_3 = nn.BatchNorm1d(layers[2])
        
        self.layer_4 = nn.Linear(layers[2],layers[3])
        torch.nn.init.xavier_uniform_(self.layer_4.weight)
        self.batch_norm_4 = nn.BatchNorm1d(layers[3])
        
        self.layer_5 = nn.Linear(layers[3],action_size)
        self.batch_norm_5 = nn.BatchNorm1d(layers[3])
    
    
        
    def forward(self, state):
        x = F.relu(self.batch_norm_1(self.layer_1(state)))
        x = F.relu(self.batch_norm_2(self.dropout_2(self.layer_2(x))))
        x = F.relu(self.batch_norm_3(self.dropout(self.layer_3(x))))
        x = F.relu(self.batch_norm_4(self.layer_4(x)))
        action = torch.tanh(self.batch_norm_5(self.layer_5(x))) * ACTION_
        return action

    
