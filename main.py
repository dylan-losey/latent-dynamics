from model import CAE
from environment import Environment
import torch
import torch.nn as nn
import torch.optim as optim
import random
import pickle
import numpy as np


def main():

    env = Environment()
    model = CAE()
    savename = "CAE_model.pt"

    EPOCHS = 40000
    HORIZON = 10
    LR = 0.001
    LR_STEP_SIZE = 1000
    LR_GAMMA = 0.5

    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA)
    traj = torch.zeros(30)


    for epoch in range(EPOCHS):

        optimizer.zero_grad()
        loss = 0.0
        thetas = torch.zeros(HORIZON)

        for task in range(HORIZON):

            thetas[task] = env.theta

            z = model.encoder(traj)
            st = env.s

            r_hat = torch.zeros(10)
            xi = torch.zeros(10,2)
            r = torch.zeros(10)

            for timestep in range(10):
                at, _, _ = model.policy(torch.cat((st, z), 0))
                rt_hat = model.decoder(torch.cat((st, z, at), 0))
                st, rt = env.dynamics(at)

                r_hat[timestep] = rt_hat
                xi[timestep,:] = st
                r[timestep] = rt

            traj = torch.cat((xi.view(-1), r), 0)

            loss += torch.norm(r_hat - r) + 0.2 * torch.norm(r_hat)

            env.reset()

        if epoch % 100 < 1:
            print(epoch)
            print(r_hat)
            print(r)
            print(thetas * 180 / np.pi)
            torch.save(model.state_dict(), savename)

        loss.backward()
        optimizer.step()
        scheduler.step()
        print(epoch, loss.item())




if __name__ == "__main__":
    main()
