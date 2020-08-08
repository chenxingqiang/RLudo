import torch


class IvanPesic(object):

    def __init__(self, env):
        self.env = env
        return

    def play(self, _, num_actions):
        # Garantovani sućeso
        # self.smart_random_move(num_actions)

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
            if self.env.dolaze_po_mene(action):
                return action

        # GET NEW ON 6
        if self.env.roll == self.env.dice_max() - 1:
            if passes[-1] == 0:
                return actions[-1]

        # ATTACK HEURISTIC
        for action in actions:
            if self.env.ilja_mode(action):
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
