from players.NeuralNets.reinforce import ReinforceAgent


class ReinforcePlayer(object):

    def __init__(self, env):
        self.log_probs = []
        self.rewards = []
        self.agent = ReinforceAgent(state_size=env.env_space(),
                                    action_size=env.action_space())
        return

    def play(self, state, num_actions):

        action, log_prob = self.agent.get_action(state)
        self.log_probs.append(log_prob)

        return action

    def recalculate(self, nextstate, reward):
        self.rewards.append(reward)
        return
