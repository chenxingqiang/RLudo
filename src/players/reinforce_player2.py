from players.NeuralNets.reinforce2 import ReinforceAgent2


class ReinforcePlayer(object):

    def __init__(self, env):
        self.log_probs = []
        self.rewards = []
        self.agent = ReinforceAgent2(state_size=env.env_space(),
                                     action_size=env.action_space())
        return

    def play(self, state, num_actions):
        return self.agent.get_action(state)

    def recalculate_step(self, _, reward):
        """
        Just stores reward for future recalculation
        :param _:
        :param reward:
        :return:
        """
        self.agent.model.reward_episode.append(reward)
        return

    def recalculate_end(self):
        """
        Once game is over recalculate everything
        :return:
        """
        self.agent.backward()
        return
