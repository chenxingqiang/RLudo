import random
import torch

class IvanPesic(object):

    def __init__(self, env):
        self.env = env
        return

    def play(self, _, num_actions):
        # Garantovani sućeso
        # self.smart_random_move(num_actions)

        # Heuristics time
        actions = [i for i in range(num_actions) if self.env.is_action_valid(i)]
        if not actions:
            return 0  # Demit Pešiću
        passes = [self.env.passed[self.env.current_player, i] for i in actions]

        # Good ol'e bubble
        for i in range(len(actions)):
            for j in range(i, len(actions)):
                if passes[i] < passes[j]:
                    tmp = passes[i]
                    passes[i] = passes[j]
                    passes[j] = tmp
                    tmp = actions[i]
                    actions[i] = actions[j]
                    actions[j] = tmp


        # DEFENSE HEURISTIC
        for action in actions:
            if self.dolaze_po_mene(action):
                return action

        # GET NEW ON 6
        if self.env.roll == self.env.dice_max() - 1:
            if passes[-1] == 0:
                return actions[-1]

        # ATTACK HEURISTIC
        for action in actions:
            if self.ilja_mode(action):
                return action

        # FAST HEURISTIC
        return actions[0]

    def recalculate_step(self, *args):
        return

    def recalculate_end(self):
        return

    def reset(self):
        return

    def smart_random_move(self, num_actions):
        for i in torch.randperm(num_actions):
            if self.env.is_action_valid(i):
                return i
        return 0  # Demit Pešiću

    def dolaze_po_mene(self, action):
        """
        Ispitujemo da li je pozicija napadnuta
        """
        my_pos = self.env.positions[self.env.current_player, action]
        if my_pos == -1:
            return False
        for pos in range(my_pos - 6, my_pos):
            player_at_pos = self.env.board_state[pos]
            if player_at_pos != 0 and player_at_pos != self.env.current_player + 1:
                return True
        return False

    def dolazice_po_mene(self, action):
        """
        Ispitujemo da li je pozicija napadnuta
        """
        my_pos = self.env.positions[self.env.current_player, action]
        if my_pos == -1:
            my_pos = self.env.starts[self.env.current_player]
        my_pass = self.env.passed[self.env.current_player, action]
        my_pos = my_pass + self.env.roll + 1
        if my_pos > self.env.board_length:
            return False
        for pos in range(my_pos - 6, my_pos):
            player_at_pos = self.env.board_state[pos]
            if player_at_pos != 0 and player_at_pos != self.env.current_player + 1:
                return True
        return False

    def ilja_mode(self, action):
        my_pos = self.env.positions[self.env.current_player, action]
        if my_pos == -1:
            if self.env.roll != self.env.dice_max() - 1:
                return False
            x = self.env.board_state[self.env.starts[self.env.current_player]]
            return x != 0 and x != self.env.current_player + 1
        s = self.env.board_state[(self.env.roll + 1 + my_pos) % self.env.board_length()]
        return s != 0 and s != self.env.current_player + 1