import random
import curses

class HumanPlayer(object):

    def __init__(self,env):
        self.env=env
        return

    def play(self, c):
        actions = [i for i in range(4) if self.env.is_action_valid(i)]
        if c==ord('0'): return 5
        if c-ord('1') in actions: return c-ord('1')
        if not actions: return 0
        return -1

    def recalculate(self, *args):
        return
