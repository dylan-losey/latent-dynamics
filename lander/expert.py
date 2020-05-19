import gym
import torch
import numpy as np
from dqn import QNetwork, ReplayBuffer


def main():

    env = gym.make('LunarReacher-v2')

    latent_size = 2
    qnetwork = QNetwork(state_size=8, latent_size=latent_size, action_size=4, seed=0)
    qnetwork.load_state_dict(torch.load('models/lander_dqn.pth'))
    qnetwork.eval()
    softmax = torch.nn.Softmax(dim=1)
    memory = ReplayBuffer(buffer_size=100, seed=0)

    episodes = 20
    scores = []

    for episode in range(episodes):

        state = env.reset()
        z = torch.FloatTensor([0.0]*latent_size)
        score = 0

        for t in range(1000):

            with torch.no_grad():
                state_t = torch.from_numpy(state).float().unsqueeze(0)
                action_values = qnetwork(z, state_t)
                action_values = softmax(action_values).cpu().data.numpy()[0]
            action = np.argmax(action_values)

            env.render()
            next_state, reward, done, info = env.step(action)
            memory.add(state, action, reward, next_state, done, info)
            state = next_state
            score += reward
            if done:
                memory.push()
                z = torch.FloatTensor(info)
                # last_traj_index = len(memory) - 1
                # states, _, rewards, _, _, _ = memory.sample_last(last_traj_index)
                # z = qnetwork.encode(states, rewards).detach()
                print(z)
                break

        scores.append(score)
        print(score)

    env.close()
    print(scores)


if __name__ == "__main__":
    main()
