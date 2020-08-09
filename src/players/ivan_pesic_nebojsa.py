import random
import torch

class IvanPesicNebojsa(object):

    def __init__(self, env):
        self.env = env
        return

    def play(self, _, num_actions):
        for i in torch.randperm(num_actions):
            if self.env.is_action_valid(i):
                return i
        return 0  # Demit Pešiću
                

    def recalculate_step(self, *args):
        return

    def recalculate_end(self,*args):
        return

    def reset(self):
        return