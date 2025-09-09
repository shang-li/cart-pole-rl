import torch
import torch.nn as nn
import torch.nn.functional as F


class PolicyNet(nn.Module):
    def __init__(self, in_dim, hidden_dim, out_dim):
        super().__init__()
        self.fc1 = nn.Linear(in_features=in_dim, out_features=hidden_dim)
        self.fc2 = nn.Linear(in_features=hidden_dim, out_features=out_dim)
    
    def forward(self, x):
        layer1 = F.relu(self.fc1(x))
        layer2 = self.fc2(layer1)
        return layer2
        
