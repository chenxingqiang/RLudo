import random
import torch
from players.NeuralNets.dqn import DQNAgent

GAMMA = 0.9

# Epsilon
EPSILON_START = 0.3
EPSILON_DECAY = 1.0

# TODO: Ovaj epsilon uopšte ne treba ovde da bude (valjda?)


class RLBasicPlayer(object):

    def __init__(self, env, read = None):
        self.agent = DQNAgent(state_size=env.env_space(),
                              action_size=env.action_space())
        if not read == None:
            self.agent.model.load_state_dict(torch.load(read))
            self.agent.model.eval()
        self.epsilon = EPSILON_START

    def play(self, state, num_actions):
        self.qvalue = self.agent.forward(state)
        greedy = torch.argmax(self.qvalue)
        if random.random() < self.epsilon:
            self.action = random.randrange(0, num_actions)
        else:
            self.action = greedy
        self.epsilon = self.epsilon * EPSILON_DECAY
        return self.action

    def recalculate_step(self, nextstate, reward):
        with torch.no_grad():
            future = self.agent.forward(nextstate)
            target = torch.tensor(self.qvalue)
            target[self.action] = reward + GAMMA * torch.max(future)
        self.agent.backward(self.qvalue, target)

    def recalculate_end(self, _):

        return

    def reset(self):
        return

    def save(self,path):
        torch.save(self.agent.model.state_dict(), path)