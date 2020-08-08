from players.NeuralNets.gcn import GraphAgent
import torch


class GraphPlayer(object):

    def __init__(self, env, read=None):
        self.log_probs = []
        self.rewards = []
        self.env = env
        self.agent = GraphAgent(env.board_length(), env.num_players, env.action_space())
        if read is not None:
            self.agent.model.load_state_dict(torch.load(read))
            self.agent.model.eval()
        return

    def play(self, _, num_actions):

        action, log_prob = self.agent.get_action(self.env.features_matrix(), self.env.adj)
        self.log_probs.append(log_prob)

        return action

    def recalculate_step(self, _, reward):
        """
        Just stores reward for future recalculation
        :param _:
        :param reward:
        :return:
        """
        self.rewards.append(reward)
        return

    def recalculate_end(self, reward):
        """
        Once game is over recalculate everything
        :return:
        """
        if reward:
            self.rewards.append(reward)
        self.agent.backward(self.rewards, self.log_probs)
        return

    def reset(self):
        self.log_probs = []
        self.rewards = []
        return

    def save(self, path):
        torch.save(self.agent.model.state_dict(), path)
