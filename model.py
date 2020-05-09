import torch
import torch.nn as nn
import torch.optim as optim
import random
import pickle
import numpy as np



class CAE(nn.Module):

    def __init__(self):
        super(CAE, self).__init__()

        self.encoder = Encoder()
        self.policy = GaussianPolicy()
        self.decoder = Decoder()


# sequence of states and rewards to z
class Encoder(nn.Module):

    def __init__(self):
        super(Encoder, self).__init__()

        self.fc1 = nn.Linear(30,30)
        self.fc2 = nn.Linear(30,1)

    def forward(self, x):
        h1 = torch.tanh(self.fc1(x))
        return self.fc2(h1)


# z and s to an action
class GaussianPolicy(nn.Module):

    def __init__(self):
        super(GaussianPolicy, self).__init__()

        self.LOG_SIG_MAX = 2
        self.LOG_SIG_MIN = -20
        self.ACTION_SCALE = 0.2

        self.fc1 = nn.Linear(3,10)
        self.fc2 = nn.Linear(10,10)
        self.mean_linear = nn.Linear(10,2)
        self.log_std_linear = nn.Linear(10,2)
        self.relu = nn.ReLU()

    def forward(self, x):
        h1 = self.relu(self.fc1(x))
        h2 = self.relu(self.fc2(h1))
        mean = self.mean_linear(h2)
        log_std = self.log_std_linear(h2)
        log_std = torch.clamp(log_std, self.LOG_SIG_MIN, self.LOG_SIG_MAX)
        action = mean + torch.randn_like(log_std) * torch.exp(log_std)
        action = self.ACTION_SCALE * torch.tanh(action)
        return action, mean, log_std


# z, state, and action to a predicted reward
class Decoder(nn.Module):

    def __init__(self):
        super(Decoder, self).__init__()

        self.fc1 = nn.Linear(5,10)
        self.fc2 = nn.Linear(10,10)
        self.r_linear = nn.Linear(10,1)
        self.relu = nn.ReLU()

    def forward(self, x):
        h1 = self.relu(self.fc1(x))
        h2 = self.relu(self.fc2(h1))
        return self.r_linear(h2)
