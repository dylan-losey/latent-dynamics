import torch
import torch.nn as nn
import torch.optim as optim
import random
import pickle


class CAE(nn.Module):

    def __init__(self):
        super(CAE, self).__init__()
        # encoder
        self.fc1 = nn.Linear(10*3,30)
        self.fc2 = nn.Linear(30,30)
        self.fc3 = nn.Linear(30,1)
        # decoder
        self.fc4 = nn.Linear(10*2+1,30)
        self.fc5 = nn.Linear(30,30)
        self.fc6 = nn.Linear(30,10)
        # loss
        self.loss_func = nn.MSELoss()

    def encoder(self, x):
        h1 = torch.tanh(self.fc1(x))
        h2 = torch.tanh(self.fc2(h1))
        return self.fc3(h2)

    def decoder(self, z_with_context):
        h4 = torch.tanh(self.fc4(z_with_context))
        h5 = torch.tanh(self.fc5(h4))
        return self.fc6(h5)

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

    def forward(self, traj):
        x, context, r_star = self.process(traj)
        z = self.encoder(x)
        z_with_context = torch.cat((z, context), 0)
        r_pred = self.decoder(z_with_context)
        loss = self.loss(r_pred, r_star)
        return loss

    def loss(self, predicted, actual):
        return self.loss_func(predicted, actual)



def main():

    EPOCH = 10000
    LR = 0.01
    LR_STEP_SIZE = 2000
    LR_GAMMA = 0.1

    model = CAE()
    dataname = "dataset.pkl"
    savename = "CAE_model.pt"

    optimizer = optim.Adam(model.parameters(), lr=LR)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=LR_STEP_SIZE, gamma=LR_GAMMA)
    data = pickle.load(open(dataname, "rb"))

    for epoch in range(EPOCH):
        random.shuffle(data)
        optimizer.zero_grad()
        loss = 0.0
        for traj in data:
            loss += model(traj)
        loss.backward()
        optimizer.step()
        scheduler.step()
        print(epoch, loss.item())
        torch.save(model.state_dict(), savename)


if __name__ == "__main__":
    main()
