import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
from players.NeuralNets.GraphConvolutionLayer import GraphConvolution

DROPOUT = 0.5
LR = 0.003
GAMMA = 0.9

class GCN(nn.Module):
    def __init__(self, nfeat, nclass):
        super(GCN, self).__init__()

        h = 8
        self.gc1 = GraphConvolution(nfeat, h)
        self.gc2 = GraphConvolution(h, nclass)

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, DROPOUT, training=self.training)
        x = self.gc2(x, adj)
        return F.log_softmax(x, dim=1)



class GraphAgent(object):
    def __init__(self, state_size, action_size):
        self.model = GCN(state_size, action_size)
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
