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
    modelname = 'CAE_model_baseline.pt'
    model = Model(modelname, traj_length, threshold)

    theta_0 = 0.0
    BATCH_SIZE = 40
    ROUNDS = 100
    traj_prev = None
    circle_x = np.cos(np.linspace(0,2*np.pi,30))
    circle_y = np.sin(np.linspace(0,2*np.pi,30))
    data = []

    for round in range(ROUNDS):

        # collect latents
        Z = np.zeros((BATCH_SIZE, 2))

        # initial run through
        # z = torch.tensor(0.0).view(1)
        # model.model.theta = theta_0
        # target = model.model.target()
        # r, states = model.model.critic(z, target)
        # states = states.view(traj_length*2)
        # traj_prev = torch.cat((states, r), 0)
        z = model.model.target()
        # z = model.model.encoder(traj_prev)
        loss = 0.0

        for episode in range(BATCH_SIZE):
            Z[episode,:] = z.detach().numpy()
            # generate policy based on z
            target = model.model.target()
            r, states = model.model.critic(z, target)
            loss += torch.norm(r).item()
            # plot the task
            # plt.plot(circle_x,circle_y,'k--')
            # plt.plot(target[0], target[1], 'ro')
            # prepare for next round
            model.model.update_target(states)
            traj = states.detach().numpy()
            states = states.view(traj_length*2)
            traj_prev = torch.cat((states, r), 0)
            # latent dynamics
            z = model.model.target()
            # z = model.model.encoder(traj_prev)

            # visualize the results
            # print("episode number: " + str(episode) + ", latent z: " + str(z.detach().numpy()))
            # plt.plot(-0.5, 0.5, 'bs')
            # plt.plot(traj[:,0], traj[:,1], 'bo')
            # plt.axis([-1.5, 1.5, -1.25, 1.25])
            # plt.savefig("figures/latent_" + str(episode))
            # plt.cla()
            # plt.show()

        # report loss
        print("loss is: ", loss)
        data.append((loss, Z))

        # visualize the latents
        # plt.plot(np.linspace(1,BATCH_SIZE,BATCH_SIZE), Z[:,0], 'bx-')
        # plt.plot(np.linspace(1,BATCH_SIZE,BATCH_SIZE), Z[:,1], 'rx-')
        # plt.show()

    print(data)
    pickle.dump(data, open("results/baseline.pkl", "wb" ))

if __name__ == "__main__":
    main()
