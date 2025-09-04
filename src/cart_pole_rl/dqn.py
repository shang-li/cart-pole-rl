import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class DQN(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.fc1 = nn.Linear(in_features=input_dim, out_features=hidden_dim, dtype=torch.float32)
        self.fc2 = nn.Linear(in_features=hidden_dim, out_features=output_dim, dtype=torch.float32)
    
    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x)))