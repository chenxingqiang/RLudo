import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable

# Constants
from torch.distributions import Categorical

GAMMA = 0.9
LR = 0.003


class ReinforceNet2(nn.Module):
    def __init__(self, state_space, action_space):
        super(ReinforceNet2, self).__init__()
        self.state_space = state_space
        self.action_space = action_space

        self.l1 = nn.Linear(self.state_space, 128, bias=False)
        self.l2 = nn.Linear(128, self.action_space, bias=False)

        self.gamma = GAMMA

        # Episode policy and reward history
        self.policy_history = Variable(torch.Tensor())
        self.reward_episode = []
        # Overall reward and loss history
        self.reward_history = []
        self.loss_history = []

    def forward(self, x):
        model = torch.nn.Sequential(
            self.l1,
            nn.Dropout(p=0.6),
            nn.ReLU(),
            self.l2,
            nn.Softmax(dim=-1)
            )
        return model(x)


class ReinforceAgent2(object):

    def __init__(self, state_size, action_size):
        self.model = ReinforceNet2(state_size, action_size)
        self.num_actions = action_size
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LR)

    def forward(self, x):
        return self.model(x)

    def backward(self):
        R = 0
        rewards = []

        # Discount future rewards back to the present using gamma
        for r in self.model.reward_episode[::-1]:
            R = r + self.model.gamma * R
            rewards.insert(0, R)

        # Scale rewards
        rewards = torch.FloatTensor(rewards)
        rewards = (rewards - rewards.mean()) / (rewards.std() + np.finfo(np.float32).eps)

        # Calculate loss
        loss = (torch.sum(torch.mul(self.model.policy_history, Variable(rewards)).mul(-1), -1))

        # Update network weights
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Save and intialize episode history counters
        self.model.loss_history.append(loss.data[0])
        self.model.reward_history.append(np.sum(self.model.reward_episode))
        self.model.policy_history = Variable(torch.Tensor())
        self.model.reward_episode = []

    def get_action(self, state):
        # Select an action (0 or 1) by running policy model and choosing based on the probabilities in state
        state = self.model(Variable(state))
        c = Categorical(state)
        action = c.sample()

        # Add log probability of our chosen action to our history
        if self.model.policy_history.dim() != 0:
            self.model.policy_history = torch.cat(
                [self.model.policy_history,
                 c.log_prob(action)])
        else:
            self.model.policy_history = (c.log_prob(action))
        return action
