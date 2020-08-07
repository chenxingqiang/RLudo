import random
import curses

class RandomPlayer(object):

    def __init__(self, seed=None):
        if seed:
            self.random = random.Random(seed)
        else:
            self.random = random.Random()
        return

    def play(self, _, num_actions):
        # TODO: Curses da queryuje o potezu
        return self.random.randrange(0, num_actions)

    def recalculate(self, *args):
        return
