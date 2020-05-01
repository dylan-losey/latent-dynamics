import torch
import torch.nn as nn
import torch.optim as optim
import random
import pickle
import numpy as np



class Circle(object):

    def __init__(self, traj_length):
        self.traj_length = traj_length

    def target_dynamics(self, episode):
        radius = 1.0
        theta = 2 * np.pi/20 * episode
        x = radius * np.cos(theta)
        y = radius * np.sin(theta)
        return np.array([x, y])

    def reward(self, s, target):
        dist2target = np.linalg.norm(s - target)
        return -1.0 * dist2target**2

    def sample(self):
        xi = []
        for idx in range(self.traj_length):
            radius = np.random.random() + 0.5
            theta = np.random.random() * 2 * np.pi
            x = radius * np.cos(theta)
            y = radius * np.sin(theta)
            xi.append([x, y])
        return xi

    def get_trajectory(self, episode):
        target = self.target_dynamics(episode)
        xi = self.sample()
        traj = []
        for s in xi:
            r = self.reward(s, target)
            traj.append((s, r, target))
        return traj


class CAE(nn.Module):

    def __init__(self):
        super(CAE, self).__init__()
        # encoder
        self.fc1 = nn.Linear(10*3,60)
        self.fc2 = nn.Linear(60,60)
        self.fc3 = nn.Linear(60,30)
        self.fc4 = nn.Linear(30,1)
        # decoder
        self.fc5 = nn.Linear(2+1,10)
        self.fc6 = nn.Linear(10,10)
        self.fc7 = nn.Linear(10,5)
        self.fc8 = nn.Linear(5,1)
        # loss
        self.loss_func = nn.MSELoss()
        self.relu = nn.ReLU()

    def encoder(self, x):
        h1 = self.relu(self.fc1(x))
        h2 = self.relu(self.fc2(h1))
        h3 = self.relu(self.fc3(h2))
        return self.fc4(h3)

    def decoder(self, z_with_context):
        h5 = self.relu(self.fc5(z_with_context))
        h6 = self.relu(self.fc6(h5))
        h7 = self.relu(self.fc7(h6))
        return self.fc8(h7)

    def process(self, traj):
        states = [0] * len(traj) * 2
        rewards = [0] * len(traj)
        for timestep, item in enumerate(traj):
            state, reward = item[0], item[1]
            states[2*timestep] = state[0]
            states[2*timestep+1] = state[1]
            rewards[timestep] = reward
        x = torch.tensor(states + rewards)
        context = torch.tensor(states)
        r_star = torch.tensor(rewards)
        return x, context, r_star

    def predict_latent(self, traj):
        x, context, r_star = self.process(traj)
        return self.encoder(x)

    def recover_reward(self, z, traj):
        r_pred = torch.zeros(len(traj), 1)
        r_star = torch.zeros(len(traj), 1)
        for timestep, item in enumerate(traj):
            state = torch.tensor(item[0])
            z_with_context = torch.cat((z, state), 0)
            r_pred[timestep,:] = self.decoder(z_with_context)
            r_star[timestep,:] = item[1]
        return r_pred, r_star

    def loss(self, r_pred, r_star):
        return self.loss_func(r_pred, r_star)


def main():

    EPOCH = 40000
    BATCH_SIZE = 10
    LR = 0.01
    LR_STEP_SIZE = 8000
    LR_GAMMA = 0.75

    traj_length = 10
    model = CAE()
    env = Circle(traj_length)
    savename = "CAE_model_1.pt"

    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA)

    for epoch in range(EPOCH):
        optimizer.zero_grad()
        loss = 0.0
        for episode in range(BATCH_SIZE):
            traj0 = env.get_trajectory(episode)
            traj1 = env.get_trajectory(episode+1)
            z = model.predict_latent(traj0)
            r_pred, r_star = model.recover_reward(z, traj1)
            loss += model.loss(r_pred, r_star)
        loss.backward()
        optimizer.step()
        scheduler.step()
        print(epoch, loss.item())
        torch.save(model.state_dict(), savename)


if __name__ == "__main__":
    main()
