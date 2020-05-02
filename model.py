import torch
import torch.nn as nn
import torch.optim as optim
import random
import pickle
import numpy as np



class CAE(nn.Module):

    def __init__(self, traj_length, threshold):
        super(CAE, self).__init__()
        # hyperparams
        self.traj_length = traj_length
        self.threshold = threshold
        # encoder
        self.e1 = nn.Linear(traj_length*3,30)
        self.e2 = nn.Linear(30,30)
        self.e3 = nn.Linear(30,1)
        self.relu = nn.ReLU()
        # decoder
        self.d1 = nn.Linear(2+1,10)
        self.d2 = nn.Linear(10,10)
        self.d3 = nn.Linear(10,2)
        self.d4 = nn.Linear(10,2)

    def target(self, episode):
        radius = 1.0
        theta = 2 * np.pi/20 * episode
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        return torch.tensor([x, y])

    def encoder(self, x):
        h1 = self.relu(self.e1(x))
        h2 = self.relu(self.e2(h1))
        return self.e3(h2)

    def decoder(self, s, z):
        z_with_context = torch.cat((z, s), 0)
        h1 = self.relu(self.d1(z_with_context))
        h2 = self.relu(self.d2(h1))
        mu = self.threshold * torch.tanh(self.d3(h2))
        log_var = self.d4(h2)
        return self.reparam(mu, log_var)

    def reparam(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def critic(self, z, target):
        r = torch.zeros(self.traj_length)
        states = torch.zeros(self.traj_length, 2)
        s = torch.tensor([0.5, 0.5])
        for timestep in range(self.traj_length):
            s += self.decoder(s, z)
            r[timestep] = -torch.norm(target - s)**2
            state = s.detach()
            states[timestep, 0] = state[0]
            states[timestep, 1] = state[1]
        return r, states


def main():

    EPOCH = 40000
    BATCH_SIZE = 10
    LR = 0.01
    LR_STEP_SIZE = 1000
    LR_GAMMA = 0.5

    traj_length = 10
    threshold = 0.2
    traj_prev = None
    savename = "CAE_model_1.pt"
    model = CAE(traj_length, threshold)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA)

    for epoch in range(EPOCH):
        optimizer.zero_grad()
        loss = 0.0
        z = torch.tensor(1.0).view(1)
        for episode in range(BATCH_SIZE):
            # latent dynamics
            if episode:
                z = model.encoder(traj_prev)
            target = model.target(episode+1)
            # generate policy based on z
            r, states = model.critic(z, target)
            states = states.view(traj_length*2)
            loss += torch.norm(r)
            # prepare for next round
            traj_prev = torch.cat((states, r.detach()), 0)
        loss.backward()
        optimizer.step()
        scheduler.step()
        print(epoch, loss.item())
        torch.save(model.state_dict(), savename)


if __name__ == "__main__":
    main()
