import numpy as np
import torch


# inside / outside environment
class Environment(object):

    def __init__(self):
        self.radius = 1.0
        self.theta = 0.0
        self.increment = np.pi/10
        self.goal = self.target()
        self.s = torch.FloatTensor([-0.5, 0.5])

    # resets agent to start, updates target position
    def reset(self):
        if torch.norm(self.s) > self.radius:
            self.theta += self.increment
        else:
            self.theta -= self.increment
        self.theta = self.theta % (2 * np.pi)
        self.goal = self.target()
        self.s = torch.FloatTensor([-0.5, 0.5])

    # target position on circle
    def target(self):
        x = self.radius * np.cos(self.theta)
        y = self.radius * np.sin(self.theta)
        return torch.FloatTensor([x, y])

    # returns next state and reward
    # detached from model
    def dynamics(self, action):
        self.s += action.detach()
        r = -torch.norm(self.goal - self.s)**2
        return self.s, r
