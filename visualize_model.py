import torch
import numpy as np
from model import(
    Circle,
    CAE
)
import pickle
import random
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import cm



def plot_reward_z(model, z, target, prev_target):

    s1 = np.linspace(-1.5, 1.5, 21)
    s2 = np.linspace(-1.5, 1.5, 21)
    X, Y = np.meshgrid(s1, s2)
    Z = np.zeros(np.shape(X))

    for idx in range(len(s1)):
        for jdx in range(len(s2)):
            x = s1[idx]
            y = s2[jdx]
            state = torch.tensor([x, y])
            z_with_context = torch.cat((z, state), 0)
            r_pred = model.model.decoder(z_with_context)
            Z[jdx, idx] = r_pred

    theta = np.linspace(0, np.pi, 11)
    plt.contourf(X, Y, Z, cmap=cm.coolwarm)
    plt.plot(np.cos(theta), np.sin(theta), 'ko-')
    plt.plot(prev_target[0], prev_target[1], 'wo')
    plt.plot(target[0], target[1], 'bo')
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.axis("equal")
    plt.show()



class Model(object):

    def __init__(self, modelname):
        self.model = CAE()
        model_dict = torch.load(modelname, map_location='cpu')
        self.model.load_state_dict(model_dict)
        self.model.eval


def main():

    traj_length = 10
    env = Circle(traj_length)
    modelname = 'CAE_model.pt'
    model = Model(modelname)

    BATCH_SIZE = 10

    for episode in range(BATCH_SIZE):
        traj0 = env.get_trajectory(episode)
        traj1 = env.get_trajectory(episode+1)
        z = model.model.predict_latent(traj0)

        target0 = traj0[0][2]
        target1 = traj1[0][2]
        plot_reward_z(model, z, target1, target0)

        r_pred0, r_star0 = model.model.recover_reward(z, traj0)
        r_pred1, r_star1 = model.model.recover_reward(z, traj1)
        r_pred0, r_star0 = r_pred0.detach().numpy(), r_star0.numpy()
        r_pred1, r_star1 = r_pred1.detach().numpy(), r_star1.numpy()
        d0 = np.linalg.norm(r_pred0 - r_star0)
        d1 = np.linalg.norm(r_pred1 - r_star1)

        # s_star, r_max = predicted_target(model, env, z, episode)

        z = z.item()
        print(episode, z, d0, d1)



if __name__ == "__main__":
    main()
