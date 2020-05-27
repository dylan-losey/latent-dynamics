import gym
import torch
import numpy as np
import sys
from dqn import QNetwork, ReplayBuffer


def main():

    env = gym.make('LunarReacher-v2')
    type = sys.argv[1]

    if type == "naive" or type == "dqn":
        latent_size = 2
    elif type == "ours":
        latent_size = 8
    qnetwork = QNetwork(state_size=8, latent_size=latent_size, action_size=4, seed=0)
    qnetwork.load_state_dict(torch.load("models/lander_" + type + ".pth"))
    qnetwork.eval()
    softmax = torch.nn.Softmax(dim=1)
    memory = ReplayBuffer(buffer_size=10, seed=0)

    episodes = 100
    scores = []
    z = torch.FloatTensor([0.0]*latent_size)

    for episode in range(episodes):

        state = env.reset()
        score = 0

        for t in range(1000):

            with torch.no_grad():
                state_t = torch.from_numpy(state).float().unsqueeze(0)
                action_values = qnetwork(z, state_t)
                action_values = softmax(action_values).cpu().data.numpy()[0]
            action = np.argmax(action_values)

            if episode > 90:
                env.render()
            next_state, reward, done, info = env.step(action)
            memory.add(state, action, reward, next_state, done, info)
            state = next_state
            score += reward
            if done:
                last_traj_index = len(memory) - 1
                states, _, rewards, _, _, _ = memory.sample_last(last_traj_index)
                if type == "dqn":
                    z = torch.FloatTensor(info)
                if type == "ours":
                    z = qnetwork.encode(states, rewards).detach()
                print(z)
                break

        scores.append(score)
        print(score)

    env.close()
    print(scores)
    print("mean score: ", np.mean(np.array(scores)))
    print("std score: ", np.std(np.array(scores)))


if __name__ == "__main__":
    main()
