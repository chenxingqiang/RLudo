import random
import curses

class HumanPlayer(object):

    def __init__(self,env):
        self.env=env
        return

    def play(self, c):
        return c-ord('1')

    def recalculate(self, *args):
        return
