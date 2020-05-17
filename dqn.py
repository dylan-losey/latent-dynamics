import numpy as np
import random
from collections import namedtuple, deque
from environment import Environment
import matplotlib.pyplot as plt
import pickle

import gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
GAMMA = 0.90            # discount factor
TAU = 0.5               # for soft update of target parameters
LR = 1e-3               # learning rate
UPDATE_EVERY = 5        # how often to update the network
TRAJ_LEN = 15           # how many timesteps in trajectory

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, latent_size, action_size, seed, fc1_units=64, fc2_units=64, enc_units=64):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            latent_size (int): Dimension of latent space
            action_size (int): Dimension of each action
            seed (int): Random seed
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
            enc_units (int): Number of nodes in encoder hidden layers
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        # Q function -> z, s to Q(z, s, a)
        self.fc1 = nn.Linear(state_size + latent_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        # Encoder -> (states, rewards) to z
        self.fc4 = nn.Linear(TRAJ_LEN*(state_size+1), enc_units)
        self.fc5 = nn.Linear(enc_units, enc_units)
        self.fc6 = nn.Linear(enc_units, latent_size)

    def encode(self, state, reward):
        context = torch.cat((state.view(-1), reward.view(-1)), 0)
        h1 = torch.tanh(self.fc4(context))
        h2 = torch.tanh(self.fc5(h1))
        return self.fc6(h2)

    def forward(self, z, state):
        z = z * torch.ones(len(state)).view(-1,1)
        context = torch.cat((z, state), 1)
        x = F.relu(self.fc1(context))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, latent_size, action_size, seed):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            latent_size (int): dimension of each latent state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.latent_size = latent_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # Q-Network
        self.qnetwork_local = QNetwork(state_size, latent_size, action_size, seed).to(device)
        self.qnetwork_target = QNetwork(state_size, latent_size, action_size, seed).to(device)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)
        self.softmax = nn.Softmax(dim=1)

        # Replay memory
        self.memory = ReplayBuffer(BUFFER_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0
        # Enter the start state for next task:
        self.state_0 = torch.FloatTensor([[-0.5, 0.5]])

    def step(self, state, action, reward, next_state, done, info):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done, info)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > 2*TRAJ_LEN:
                traj_1, traj_2 = self.memory.sample()
                self.learn(traj_1, traj_2, GAMMA)

    def predict(self):
        """Predict the latent value for the next task given the current trajectory."""
        states, rewards = self.memory.last_traj()
        return self.qnetwork_target.encode(states, rewards).detach()

    def act(self, z, state, eps=0.):
        """Returns actions for given state as per current policy.

        Params
        ======
            z (torch): latent variable for task
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(z.detach(), state)
            action_values = self.softmax(action_values).cpu().data.numpy()[0]
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values)
            # return np.random.choice(np.arange(self.action_size), p=action_values)
        else:
            return random.choice(np.arange(self.action_size))


    def learn(self, traj_1, traj_2, gamma):
        """Update value parameters using given batch of experience tuples.
        Params
        ======
            traj_1, traj_2 (Tuple[torch.Tensor]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states_1, actions_1, rewards_1, next_states_1, dones_1, info_1 = traj_1
        states_2, actions_2, rewards_2, next_states_2, dones_2, info_2 = traj_2

        """Get max predicted Q values (for next states) from target model"""
        # z = info_1[-1] * 0      # for naive baseline
        # z = info_1[-1]        # for dqn baseline
        z = self.qnetwork_target.encode(states_1, rewards_1).detach()
        Q_targets_next = self.qnetwork_target(z, next_states_2).detach().max(1)[0].unsqueeze(1)
        """Get max predicted Q value for next task after reset from target model"""
        # z_reset = info_2[-1] * 0    # for naive baseline
        # z_reset = info_2[-1]      # for dqn baseline
        z_reset = self.qnetwork_target.encode(states_2, rewards_2).detach()
        Q_targets_reset = self.qnetwork_target(z_reset, self.state_0).detach().max(1)[0].unsqueeze(1)
        """Compute Q targets for current states"""
        Q_targets = rewards_2 + (gamma * Q_targets_next * (1 - dones_2)) + (0.0 * Q_targets_reset * dones_2)

        """Get expected Q values from local model"""
        # z = info_1[-1] * 0      # for naive baseline
        # z = info_1[-1]        # for dqn baseline
        z = self.qnetwork_local.encode(states_1, rewards_1)
        Q_expected = self.qnetwork_local(z, states_2).gather(1, actions_2)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        self.soft_update(self.qnetwork_local, self.qnetwork_target, TAU)

    def soft_update(self, local_model, target_model, tau):
        """Soft update model parameters.
        θ_target = τ*θ_local + (1 - τ)*θ_target
        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
            tau (float): interpolation parameter
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size, seed):
        """Initialize a ReplayBuffer object.
        Params
        ======
            buffer_size (int): maximum size of buffer
            seed (int): random seed
        """
        self.memory = deque(maxlen=buffer_size)
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done", "info"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done, info):
        """Add a new experience to memory."""
        e = self.experience(state, action, reward, next_state, done, info)
        self.memory.append(e)

    def last_traj(self):
        """Get the last trajectory from memory."""
        traj = []
        for t in range(len(self.memory) - TRAJ_LEN, len(self.memory)):
            traj.append(self.memory[t])
        states = torch.from_numpy(np.vstack([e.state for e in traj if e is not None])).float().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in traj if e is not None])).float().to(device)
        return (states, rewards)

    def sample(self):
        """Randomly sample a pair of sequential trajectories from memory."""

        start_index = random.randint(0, int(len(self.memory)/TRAJ_LEN) - 2)
        start_index *= TRAJ_LEN

        traj_1, traj_2 = [], []
        for t in range(start_index, start_index + TRAJ_LEN):
            traj_1.append(self.memory[t])
            traj_2.append(self.memory[t + TRAJ_LEN])

        states_1 = torch.from_numpy(np.vstack([e.state for e in traj_1 if e is not None])).float().to(device)
        actions_1 = torch.from_numpy(np.vstack([e.action for e in traj_1 if e is not None])).long().to(device)
        rewards_1 = torch.from_numpy(np.vstack([e.reward for e in traj_1 if e is not None])).float().to(device)
        next_states_1 = torch.from_numpy(np.vstack([e.next_state for e in traj_1 if e is not None])).float().to(device)
        dones_1 = torch.from_numpy(np.vstack([e.done for e in traj_1 if e is not None]).astype(np.uint8)).float().to(device)
        info_1 = torch.from_numpy(np.vstack([e.info for e in traj_1 if e is not None])).float().to(device)
        prev_traj = (states_1, actions_1, rewards_1, next_states_1, dones_1, info_1)

        states_2 = torch.from_numpy(np.vstack([e.state for e in traj_2 if e is not None])).float().to(device)
        actions_2 = torch.from_numpy(np.vstack([e.action for e in traj_2 if e is not None])).long().to(device)
        rewards_2 = torch.from_numpy(np.vstack([e.reward for e in traj_2 if e is not None])).float().to(device)
        next_states_2 = torch.from_numpy(np.vstack([e.next_state for e in traj_2 if e is not None])).float().to(device)
        dones_2 = torch.from_numpy(np.vstack([e.done for e in traj_2 if e is not None]).astype(np.uint8)).float().to(device)
        info_2 = torch.from_numpy(np.vstack([e.info for e in traj_2 if e is not None])).float().to(device)
        curr_traj = (states_2, actions_2, rewards_2, next_states_2, dones_2, info_2)

        return prev_traj, curr_traj

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

def train(agent,
    n_episodes=40000, max_t=15, eps_start=1.0, eps_end=0.05, eps_decay=0.999, savename="dqn.pth"):
    """Deep Q-Learning.

    Params
    ======
        agent: agent to train
        n_episodes (int): maximum number of training episodes
        max_t (int): maximum number of timesteps per episode
        eps_start (float): starting value of epsilon, for epsilon-greedy action selection
        eps_end (float): minimum value of epsilon
        eps_decay (float): multiplicative factor (per episode) for decreasing epsilon
    """
    scores = []                        # list containing scores from each episode
    thetas = []                        # list containing target position from each episode
    scores_window = deque(maxlen=100)  # last 100 scores
    eps = eps_start                    # initialize epsilon
    z = torch.FloatTensor([0.0]*agent.latent_size)
    for i_episode in range(1, n_episodes+1):
        state = env.reset()
        score = 0
        thetas.append(env.theta)
        for t in range(max_t):
            action = agent.act(z, state, eps)
            # env.render()
            next_state, reward, done, info = env.step(action)
            agent.step(state, action, reward, next_state, done, info)
            state = next_state
            score += reward
            if done:
                # z = torch.FloatTensor(info) * 0     # for naive baseline
                # z = torch.FloatTensor(info)       # for dqn baseline
                z = agent.predict()       # encode z for next task
                break
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), savename)
    torch.save(agent.qnetwork_local.state_dict(), savename)
    env.close()
    return (scores, thetas)

if __name__ == "__main__":
    env = Environment()
    agent = Agent(state_size=2, latent_size=8, action_size=5, seed=0)
    performance = train(agent, savename="models/circle_dqn.pth")
    pickle.dump(performance, open("results/ours-greedy4.pkl", "wb" ))
