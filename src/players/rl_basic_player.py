import random
import players.NN1
import torch

GAMMA = 0.9


class RLBasicPlayer(object):

    def __init__(self, env):
        self.nn = None

    def play(self, state, num_actions):
        self.qvalue = self.nn.forward(torch.from_numpy(state))
        greedy = torch.argmax(qvalue)
        if (random.random() < epsilon):
            self.action = random.randrange(0, num_actions)
        else:
            self.action = greedy
        return self.action

    def recalculate(self, nextstate, reward):
        with torch.no_grad():
            future = self.nn.forward(torch.from_numpy(nextstate))
            target = torch.tensor(self.qvalue)
            target[self.action] = reward + GAMMA * torch.max(future)
        self.nn.backward(qvalue, target)
