
import numpy as np
import random

from collections import namedtuple, deque
from replaybuffer import ExperienceReplay
from model import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim

BUFFER_SIZE = int(1e5)  # replay buffer size
BATCH_SIZE = 32         # minibatch size
GAMMA = 0.99            # discount factor
TAU = 1e-3              # for soft update of target parameters
LR = 5e-4               # learning rate
C = 1250                # how often to update the network

class Agent():
    """Interacts with and learns from the environment."""

    def __init__(self, state_size, action_size, seed, algorithm = 'DDQN'):
        """Initialize an Agent object.

        Params
        ======
            state_size (int): dimension of each state
            action_size (int): dimension of each action
            seed (int): random seed
        """
        self.state_size = state_size
        self.action_size = action_size
        self.seed = random.seed(seed)

        # algorithm
        self.algorithm = algorithm
        # Q-Network
        self.qnetwork_local = QNetwork(state_size, action_size, seed)
        self.qnetwork_target = QNetwork(state_size, action_size, seed)
        self.optimizer = optim.Adam(self.qnetwork_local.parameters(), lr=LR)

        # Replay memory
        self.memory = ExperienceReplay(action_size, BUFFER_SIZE, BATCH_SIZE, seed)
        # Initialize time step (for updating every UPDATE_EVERY steps)
        self.t_step = 0

    def step(self, state, action, reward, next_state, done):
        # Save experience in replay memory
        self.memory.add(state, action, reward, next_state, done)

        # If enough samples are available in memory, get random subset and learn
        if len(self.memory) > BATCH_SIZE:
            experiences = self.memory.sample()
            self.learn(experiences, GAMMA)

        # Update target network every C steps.
        self.t_step = (self.t_step + 1) % C
        if self.t_step == 0:
            # ------------------- update target network ------------------- #
            self.hard_update(self.qnetwork_local, self.qnetwork_target)

    def choose_action(self, state, eps=0.):
        """Returns actions for given state as per current policy.

        Params
        ======
            state (array_like): current state
            eps (float): epsilon, for epsilon-greedy action selection
        """
        # Epsilon-greedy action selection
        if random.random() < eps:
            return random.choice(np.arange(self.action_size))
        else:
            state = torch.from_numpy(state).float().unsqueeze(0)
            self.qnetwork_local.eval() # evaluation mode activated
            with torch.no_grad():
                action_values = self.qnetwork_local(state)
            self.qnetwork_local.train() # set train mode
            return np.argmax(action_values.numpy())

    def learn(self, experiences, gamma):
        """Update value parameters using given batch of experience tuples.

        Params
        ======
            experiences (Tuple[torch.Variable]): tuple of (s, a, r, s', done) tuples
            gamma (float): discount factor
        """
        states, actions, rewards, next_states, dones = experiences

        if self.algorithm == 'DDQN':
            # Double DQN with Target Networks
            values,indices = self.qnetwork_local(next_states).detach().max(1)
            Q_targets_next = self.qnetwork_target(next_states).detach().gather(1,indices.unsqueeze(1))
        elif self.algorithm == 'DQN':
            # DQN with Target Networks
            Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))
        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)
        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def hard_update(self, local_model, target_model):
        """Hard update model parameters. Copy the values of local network into the target.
        θ_target = θ_local

        Params
        ======
            local_model (PyTorch model): weights will be copied from
            target_model (PyTorch model): weights will be copied to
        """
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(local_param.data)

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
