import torch
import torch.nn as nn
import torch._dynamo
from settings import ACTION_
torch._dynamo.config.suppress_errors = True
    
class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size, 256)
        self.fc2 = nn.Linear(256, 300)
        
        self.fc3v = nn.Linear(300, 350)
        self.fc3w = nn.Linear(300, 350)
        
        self.fc4v = nn.Linear(350, 400)
        self.fc4w = nn.Linear(350, 400)
        
        self.fc5v = nn.Linear(400, 512)
        self.fc5w = nn.Linear(400, 512)
        self.bn5 = nn.LayerNorm(512)
        
        self.fc6v = nn.Linear(512, 1)
        self.fc6w = nn.Linear(512, 1)
        
    def forward(self, state):
    
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        
        v = torch.relu(self.fc3v(x))
        v = torch.relu(self.fc4v(v))
        v = torch.relu(self.fc5v(v))
        v = self.bn5(v)
        v = torch.tanh(self.fc6v(v))
        
        w = torch.relu(self.fc3w(x))
        w = torch.relu(self.fc4w(w))
        w = torch.relu(self.fc5w(w))
        w = self.bn5(w)
        w = torch.tanh(self.fc6w(w))
        
        return torch.cat([v*ACTION_, w*ACTION_], dim=-1)
    
class Critic1(nn.Module):
    def __init__(self, state_size, action_size):
        super(Critic1, self).__init__()
        self.fc1 = nn.Linear(state_size + action_size, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, 1)
        

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
    

class Critic2(nn.Module):
    def __init__(self, state_size, action_size):
        super(Critic2, self).__init__()
        self.fc1 = nn.Linear(state_size + action_size, 500)
        self.fc2 = nn.Linear(500, 512)
        self.fc3 = nn.Linear(512, 500)
        self.fc4 = nn.Linear(500, 1)
        

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)