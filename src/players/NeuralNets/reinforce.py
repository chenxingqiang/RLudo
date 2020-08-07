import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

# Constants
GAMMA = 0.9
LR = 0.003


class ReinforceNet(nn.Module):
    def __init__(self, state_size, action_size):
        super(ReinforceNet, self).__init__()

        h = 200

        self.neuralnet = nn.Sequential(
            nn.Linear(state_size, h),
            nn.ReLU(),
            nn.Linear(h, h),
            nn.ReLU(),
            nn.Linear(h, action_size))

    def forward(self, x):
        return F.softmax(self.neuralnet(x))


class ReinforceAgent(object):
    def __init__(self, state_size, action_size):
        self.model = ReinforceNet(state_size, action_size)
        self.num_actions = action_size
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=LR)

    def forward(self, x):
        return self.model(x)

    def backward(self, rewards, log_probs):
        """
        Implements the REINFORCE algorithm for policy gradient.
        :param rewards: Reward history
        :param log_probs: Log-prob history
        :return: None
        """
        discounted_rewards = []

        for t in range(len(rewards)):
            gt = 0
            pw = 0
            for r in rewards[t:]:
                gt = gt + GAMMA ** pw * r
                pw = pw + 1
            discounted_rewards.append(gt)

        discounted_rewards = torch.tensor(discounted_rewards)
        discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (
                    discounted_rewards.std() + 1e-9)  # normalize discounted rewards

        policy_gradient = []
        for log_prob, gt in zip(log_probs, discounted_rewards):
            policy_gradient.append(-log_prob * gt)

        self.optimizer.zero_grad()
        policy_gradient = torch.stack(policy_gradient).sum()
        policy_gradient.backward()
        self.optimizer.step()

    def get_action(self, state):
        """
        Runs a state through NN to get action probabilities
        :param state: Current state
        :return: Most probable action and log probability
        """
        probs = self.forward(Variable(state))
        highest_prob_action = np.random.choice(self.num_actions, p=np.squeeze(probs.detach().numpy()))
        log_prob = torch.log(probs.squeeze(0)[highest_prob_action])
        return highest_prob_action, log_prob
