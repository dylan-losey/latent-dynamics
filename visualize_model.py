import torch
import numpy as np
from model import CAE
import pickle
import random
import matplotlib.pyplot as plt



class Model(object):

    def __init__(self, modelname, traj_length, threshold):
        self.model = CAE(traj_length, threshold)
        model_dict = torch.load(modelname, map_location='cpu')
        self.model.load_state_dict(model_dict)
        self.model.eval


def main():

    traj_length = 10
    threshold = 0.2
    modelname = 'CAE_model_1.pt'
    model = Model(modelname, traj_length, threshold)

    BATCH_SIZE = 10
    traj_prev = None
    z = torch.tensor(1.0).view(1)

    for episode in range(BATCH_SIZE):
        # latent dynamics
        if episode:
            z = model.model.encoder(traj_prev)
        target = model.model.target(episode+1)
        # generate policy based on z
        r, states = model.model.critic(z, target)
        states_flat = states.view(traj_length*2)
        # prepare for next round
        traj_prev = torch.cat((states_flat, r.detach()), 0)
        traj = states.numpy()

        print(episode, z.item())
        plt.plot(traj[:,0], traj[:,1], 'ko--')
        plt.plot(target[0], target[1], 'bo')

    plt.axis("equal")
    plt.show()


if __name__ == "__main__":
    main()
