import torch
import torch.nn as nn
import torch._dynamo
torch._dynamo.config.suppress_errors = True

class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_size, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, action_size)
        
    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))*0.5
    
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
        self.fc2 = nn.Linear(500, 600)
        self.fc3 = nn.Linear(600, 300)
        self.fc4 = nn.Linear(300, 1)
        

    def forward(self, state, action):
        x = torch.cat([state, action], dim=1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        return self.fc4(x)