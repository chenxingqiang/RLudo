import random


class RandomPlayer(object):

    def __init__(self, seed = None):
        if seed:
            self.random = random.Random(seed)
        else:
            self.random = random.Random()
        return


    def play(self, state, num_actions):
        return self.random.randrange(0, num_actions)
