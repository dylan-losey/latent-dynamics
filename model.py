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
        self.theta = 0.0
        self.radius = 1.0
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

    def update_target(self, states):
        s_final = states[-1,:]
        if torch.norm(s_final) > self.radius:
            self.theta += np.pi/10
        else:
            self.theta -= np.pi/10

    def target(self):
        x = self.radius * np.cos(self.theta)
        y = self.radius * np.sin(self.theta)
        return torch.tensor([x, y])

    def encoder(self, x):
        h1 = self.relu(self.e1(x))
        h2 = self.relu(self.e2(h1))
        return self.e3(h2)

    def decoder(self, s, z):
        z_with_context = torch.cat((z, s), 0)
        h1 = self.relu(self.d1(z_with_context))
        h2 = self.relu(self.d2(h1))
        action = self.reparam(self.d3(h2), self.d4(h2))
        return self.threshold * torch.tanh(action)

    def reparam(self, mu, log_var):
        std = torch.exp(0.5*log_var)
        eps = torch.randn_like(std)
        return mu + eps * std

    def critic(self, z, target):
        r = torch.zeros(self.traj_length)
        states = torch.zeros(self.traj_length, 2)
        s = torch.tensor([-0.5, 0.5])
        for timestep in range(self.traj_length):
            s += self.decoder(s, z)
            r[timestep] = -torch.norm(target - s)**2
            states[timestep, :] = s
        return r, states


def main():

    EPOCH = 40000
    BATCH_SIZE = 20
    LR = 0.001
    LR_STEP_SIZE = 4000
    LR_GAMMA = 0.5

    traj_length = 10
    threshold = 0.2
    traj_prev = None
    savename = "CAE_model_1.pt"
    model = CAE(traj_length, threshold)

    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA)
    min_loss = np.Inf
    min_epoch = 0

    for epoch in range(EPOCH):
        optimizer.zero_grad()
        loss = 0.0
        for trial in range(10):
            # reset target position
            z = torch.tensor(1.0).view(1)
            model.theta = 0.0
            # initialize trajectory
            target = model.target()
            r, states = model.critic(z, target)
            states = states.view(traj_length*2)
            traj_prev = torch.cat((states, r), 0)
            z = model.encoder(traj_prev)
            # loop through set number of tasks
            for episode in range(BATCH_SIZE):
                # generate policy based on z
                target = model.target()
                r, states = model.critic(z, target)
                loss += torch.norm(r)
                # prepare for next round
                model.update_target(states)
                states = states.view(traj_length*2)
                traj_prev = torch.cat((states, r), 0)
                # latent dynamics
                z = model.encoder(traj_prev)
        if loss.item() < min_loss:
            min_loss = loss.item()
            min_epoch = epoch
            torch.save(model.state_dict(), savename)
        loss.backward()
        optimizer.step()
        scheduler.step()
        print(epoch, min_epoch, loss.item() / 10.0)


if __name__ == "__main__":
    main()
