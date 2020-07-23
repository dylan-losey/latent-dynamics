import numpy as np
import random
from collections import namedtuple, deque
import pickle
import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from environment import Franka

BUFFER_SIZE = int(1e3)  # replay buffer size
GAMMA = 0.99            # discount factor
TAU = 1e-2              # for soft update of target parameters
LR = 1e-3               # learning rate
UPDATE_EVERY = 4        # how often to update the network
TRAJ_LEN_Q = 40         # how many timesteps in trajectory for Q value
TRAJ_LEN_recon = 1      # how many timesteps in trajectory for enc-decoder

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


class QNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, latent_size, action_size, seed, fc1_units=64, fc2_units=64, enc_units=16, dec_units=16):
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
            dec_units (int): Number of nodes in decoder hidden layers
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        # Q function -> z, s to Q(z, s, a)
        self.fc1 = nn.Linear(state_size + latent_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        # Encoder -> (states, rewards) to z
        self.fc4 = nn.Linear(TRAJ_LEN_recon*(state_size+1), enc_units)
        self.fc5 = nn.Linear(enc_units, enc_units)
        self.fc6 = nn.Linear(enc_units, latent_size)
        # Decoder -> z, s to r
        self.fc7 = nn.Linear(latent_size+state_size, dec_units)
        self.fc8 = nn.Linear(dec_units, dec_units)
        self.fc9 = nn.Linear(dec_units, 1)

    def encode(self, state, reward):
        context = torch.cat((state.view(-1), reward.view(-1)), 0)
        h1 = F.relu(self.fc4(context))
        h2 = F.relu(self.fc5(h1))
        return self.fc6(h2)

    def decode(self, z, state):
        z = z * torch.ones(len(state), device=device).view(-1,1)
        context = torch.cat((z, state), 1)
        h1 = F.relu(self.fc7(context))
        h2 = F.relu(self.fc8(h1))
        return self.fc9(h2)

    def forward(self, z, state):
        z = z * torch.ones(len(state), device=device).view(-1,1)
        context = torch.cat((z, state), 1)
        x = F.relu(self.fc1(context))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, type, state_size, latent_size, action_size, seed):
        """Initialize an Agent object.

        Params
        ======
            type (string): is this ours, dqn, naive
            state_size (int): dimension of each state
            latent_size (int): dimension of each latent state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.type = type
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

    def step(self, state, action, reward, next_state, done, info):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done, info)

        # Learn every UPDATE_EVERY time steps.
        self.t_step = (self.t_step + 1) % UPDATE_EVERY
        if self.t_step == 0:
            # If enough samples are available in memory, get random subset and learn
            if len(self.memory) > 2:
                index = self.memory.sample_pair()
                self.learn(index, GAMMA)

    def predict(self):
        """Predict the latent value for the next task given the current trajectory."""
        last_traj_index = len(self.memory) - 1
        states, _, rewards, _, _, _ = self.memory.sample_last(last_traj_index)
        return self.qnetwork_target.encode(states, rewards).detach()

    def act(self, z, state, eps=0.0):
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
            action_values = self.qnetwork_local(z.detach().to(device), state.to(device))
            action_values = self.softmax(action_values).cpu().data.numpy()[0]
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values)
        else:
            return random.choice(np.arange(self.action_size))


    def learn(self, index, gamma):
        """Update value parameters using given batch of experience tuples.
        Params
        ======
            index: index of first trajectory in sequence
            gamma (float): discount factor
        """

        """Collect experiences from sampled sequence of trajectories."""
        states_prev, _, rewards_prev, _, _, info_prev = self.memory.sample_last(index)
        states_last, _, rewards_last, _, _, info_last = self.memory.sample_last(index+1)
        states, actions, rewards, next_states, dones, info = self.memory.sample_random(index+1)

        """Get max predicted Q values (for next states) from target model"""
        if self.type == "naive":
            z = info_prev * 0      # for naive baseline
        elif self.type == "dqn":
            z = info_prev          # for dqn baseline
        elif self.type == "ours":
            z = self.qnetwork_target.encode(states_prev, rewards_prev).detach()
        Q_targets_next = self.qnetwork_target(z, next_states).detach().max(1)[0].unsqueeze(1)
        """Compute Q targets for current states"""
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        """Get expected Q values from local model"""
        if self.type == "naive":
            z = info_prev * 0      # for naive baseline
        elif self.type == "dqn":
            z = info_prev          # for dqn baseline
        elif self.type == "ours":
            z = self.qnetwork_local.encode(states_prev, rewards_prev)
        Q_expected = self.qnetwork_local(z, states).gather(1, actions)

        pred_reward = self.qnetwork_local.decode(z, states_last)

        # Compute loss
        self.Q_loss = Q_loss = F.mse_loss(Q_expected, Q_targets)
        self.recon_loss = recon_loss = F.mse_loss(rewards_last, pred_reward)
        if self.type == "ours":
            loss = Q_loss + recon_loss
        else:
            loss = Q_loss

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
        self.trajectory = deque()
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done", "info"])
        self.seed = random.seed(seed)

    def add(self, state, action, reward, next_state, done, info):
        """Add a new experience to trajectory."""
        e = self.experience(state, action, reward, next_state, done, info)
        self.trajectory.append(e)
        if done:
            self.push()

    def push(self):
        """Add a trajectory to memory."""
        self.memory.append(self.trajectory)
        self.trajectory = deque()

    def sample_random(self, idx):
        """Get random experiences from trajectory."""
        traj = self.memory[idx]
        xi = random.sample(traj, k=TRAJ_LEN_Q)
        return self.process(xi)

    def sample_last(self, idx):
        """Get last experiences from trajectory."""
        traj = self.memory[idx]
        xi = []
        for jdx in range(len(traj)-TRAJ_LEN_recon, len(traj)):
            xi.append(traj[jdx])
        return self.process(xi)

    def process(self, traj):
        """Convert experiences to torch tensors."""
        states = torch.from_numpy(np.vstack([e.state for e in traj if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in traj if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in traj if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in traj if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in traj if e is not None]).astype(np.uint8)).float().to(device)
        info = torch.from_numpy(np.vstack([e.info for e in traj if e is not None])).float().to(device)
        return (states, actions, rewards, next_states, dones, info)

    def sample_pair(self):
        """Randomly sample trajectory from memory."""
        return random.randint(0, len(self.memory) - 2)

    def __len__(self):
        """Return the current size of internal memory."""
        return len(self.memory)

def train(agent, type,
    n_episodes=10000, max_t=1500, eps_start=1.0, eps_end=0.01, eps_decay=0.999, savename="dqn.pth"):
    """Deep Q-Learning.

    Params
    ======
        agent: agent to train
        type (string): is this naive, dqn, ours
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
            next_state, reward, done, info = env.step(action)
            agent.step(state, action, reward, next_state, done, info)
            state = next_state
            score += reward
            # env.render()
            if done:
                if type == "dqn":
                    z = torch.FloatTensor(info)     # for dqn baseline
                if type == "ours":
                    z = agent.predict()             # encode z for next task
                break
        scores_window.append(score)       # save most recent score
        scores.append(score)              # save most recent score
        eps = max(eps_end, eps_decay*eps) # decrease epsilon
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)), end="")
        if i_episode % 100 == 0:
            print('\rEpisode {}\tAverage Score: {:.2f}'.format(i_episode, np.mean(scores_window)))
            torch.save(agent.qnetwork_local.state_dict(), savename)
            print('Q Loss: {}'.format(agent.Q_loss.item()))
            print('Recon Loss: {}'.format(agent.recon_loss.item()))
    torch.save(agent.qnetwork_local.state_dict(), savename)
    env.close()
    return (scores, thetas)

if __name__ == "__main__":
    env = Franka()
    type = sys.argv[1]
    if type == "naive" or type == "dqn":
        latent_size =3
    elif type == "ours":
        latent_size = 8
    agent = Agent(type, state_size=3, latent_size=latent_size, action_size=7, seed=100)
    # agent.qnetwork_local.load_state_dict(torch.load("models/franka_test_" + type + ".pth"))
    performance = train(agent, type, savename="models/franka_test_" + type + ".pth")
    pickle.dump(performance, open("results/franka_test_" + type + ".pkl", "wb" ))
