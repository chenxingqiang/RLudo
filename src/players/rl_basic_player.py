import random
import torch
from players.dqn.dqn import DQNAgent

GAMMA = 0.9

# Epsilon
EPSILON_START = 1.0
EPSILON_DECAY = 0.95

# TODO: Ovaj epsilon decay uopšte ne treba ovde da bude


class RLBasicPlayer(object):

    def __init__(self, env):
        self.agent = DQNAgent(state_size=env.env_space(),
                              action_size=env.action_space())
        self.epsilon = EPSILON_START

    def play(self, state, num_actions):
        state = self.fix_state(state)
        self.qvalue = self.agent.forward(state)
        greedy = torch.argmax(self.qvalue)
        if random.random() < self.epsilon:
            self.action = random.randrange(0, num_actions)
        else:
            self.action = greedy
        self.epsilon = self.epsilon * EPSILON_DECAY
        return self.action

    def recalculate(self, nextstate, reward):
        nextstate = self.fix_state(nextstate)
        with torch.no_grad():
            future = self.agent.forward(nextstate)
            target = torch.tensor(self.qvalue)
            target[self.action] = reward + GAMMA * torch.max(future)
        self.agent.backward(self.qvalue, target)

    def fix_state(self, state):
        """
        Pretvara state koji je tuple u state koji je tenzor
        :param state:
        :return:
        """
        return torch.cat((state[0],
                          torch.flatten(state[1]),
                          torch.ones(1) * state[2])).float()
