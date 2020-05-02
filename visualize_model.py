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
    modelname = 'CAE_model.pt'
    model = Model(modelname, traj_length, threshold)

    BATCH_SIZE = 20
    traj_prev = None
    z = torch.tensor(1.0).view(1)
    model.model.theta = 0.0
    circle_x = np.cos(np.linspace(0,2*np.pi,30))
    circle_y = np.sin(np.linspace(0,2*np.pi,30))

    # collect latents
    Z = []

    # initialize trajectory
    target = model.model.target()
    r, states = model.model.critic(z, target)
    states = states.view(traj_length*2)
    traj_prev = torch.cat((states, r), 0)
    z = model.model.encoder(traj_prev)

    for episode in range(BATCH_SIZE):
        # generate policy based on z
        target = model.model.target()
        r, states = model.model.critic(z, target)
        # plot the task
        plt.plot(circle_x,circle_y,'k--')
        plt.plot(target[0], target[1], 'ro')
        # prepare for next round
        model.model.update_target(states)
        traj = states.detach().numpy()
        states = states.view(traj_length*2)
        traj_prev = torch.cat((states, r), 0)
        # latent dynamics
        z = model.model.encoder(traj_prev)
        Z.append(z.item())

        # visualize the results
        print("episode number: " + str(episode) + ", latent z: " + str(z.item()))
        plt.plot(-0.5, 0.5, 'bs')
        plt.plot(traj[:,0], traj[:,1], 'bo')
        plt.axis([-1.5, 1.5, -1.25, 1.25])
        plt.savefig("figures/episode_" + str(episode))
        plt.cla()
        # plt.show()


    # visualize the latents
    plt.plot(np.linspace(1,20,20), Z, 'bx-')
    plt.show()


if __name__ == "__main__":
    main()
