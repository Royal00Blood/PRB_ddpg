import torch
import torch.nn as nn
import torch._dynamo
import torch.nn.functional as F
from settings import (ACTION_, STATE_SIZE, SEED,
                      ACTION_SIZE,LAYER_A)
torch._dynamo.config.suppress_errors = False
    
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
        self.batch_norm_1 = nn.LayerNorm(layers[0])
        
        self.layer_2 = nn.Linear(layers[0],layers[1])
        self.batch_norm_2 = nn.LayerNorm(layers[1])
        
        # self.layer_3 = nn.Linear(layers[1],layers[2])
        # self.batch_norm_3 = nn.LayerNorm(layers[2])
        
        # self.layer_4 = nn.Linear(layers[2],layers[3])
        # self.batch_norm_4 = nn.LayerNorm(layers[3])
        
        self.layer_5 = nn.Linear(layers[1],action_size)
        self.reset_weights()
    
    def reset_weights(self):
        nn.init.kaiming_normal_(self.layer_1.weight, mode='fan_in',nonlinearity='relu')
        nn.init.kaiming_normal_(self.layer_2.weight, mode='fan_in',nonlinearity='relu')
        # nn.init.kaiming_normal_(self.layer_3.weight, mode='fan_in',nonlinearity='relu')
        # nn.init.kaiming_normal_(self.layer_4.weight, mode='fan_in',nonlinearity='relu')
        nn.init.xavier_uniform_(self.layer_5.weight)
        
    def forward(self, state):
        x = self.layer_1(state)
        x = self.batch_norm_1(x)
        x = self.dropout(x)
        x = F.relu(x)
        
        x = self.layer_2(x)
        x = self.batch_norm_2(x)
        x = self.dropout(x)
        x = F.relu(x)
        
        # x = self.layer_3(x)
        # #x = self.batch_norm_3(x)
        # x = self.dropout(x)
        # x = F.relu(x)
        
        # x = self.layer_4(x)
        # #x = self.batch_norm_4(x)
        # x = self.dropout(x)
        # x = F.relu(x)
        
        action = self.layer_5(x)
        action = torch.tanh(action) * ACTION_
        return action



    
