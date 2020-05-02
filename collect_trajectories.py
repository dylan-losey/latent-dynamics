import numpy as np
import pickle
import copy


def target_dynamics(episode):
    radius = 1.0
    theta = 2 * np.pi/20 * episode
    x = radius * np.cos(theta)
    y = radius * np.sin(theta)
    return np.array([x, y])

def robot_dynamics(s_0, s_star, T):
    threshold = 0.2
    target_radius = 0.1
    s, xi = copy.deepcopy(s_0), []
    delta = s_star - s
    dist2s_star = np.linalg.norm(delta)
    for timestep in range(T):
        if dist2s_star > threshold:
            delta = delta / dist2s_star * threshold
        s += delta
        delta = s_star - s
        dist2s_star = np.linalg.norm(delta)
        xi.append(copy.deepcopy(s))
    return xi

def reward(s, target):
    dist2target = np.linalg.norm(s - target)
    return -1.0 * dist2target



def main():

    EPISODES = 1000
    TIMESTEPS = 10
    s_0 = np.array([0.0, 0.0])

    data = []
    savename = "dataset.pkl"

    for episode in range(EPISODES):
        target = target_dynamics(episode)
        s_star = 3 * (np.random.random(2) - 0.5)
        xi = robot_dynamics(s_0, s_star, TIMESTEPS)
        tau = []
        for s in xi:
            r = reward(s, target)
            tau.append((s, r, target))
        data.append(tau)

    pickle.dump(data, open(savename, "wb" ))
    print("number of collected traj: ", len(data))

if __name__ == "__main__":
    main()
