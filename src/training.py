from environment import Ludo
from players.rl_basic_player import RLBasicPlayer
from players.random_player import RandomPlayer
from players.reinforce_player import ReinforcePlayer
from game_loop import loop
import random

TOKENS = 4
EPISODES = 10
PLAYERS = 4


def run_one_episode(state, agents):
    game_end = False
    env.reset(PLAYERS)
    for agent in agents:
        agent.reset()
    while not game_end:
        x = env.current_player
        action = agents[x].play(state, TOKENS)
        nextstate, reward, game_end = env.step(action)
        agents[x].recalculate_step(nextstate, reward)
        state = nextstate
    for agent in agents:
        agent.recalculate_end()


if __name__ == '__main__':

    #Initialize environment
    env = Ludo(PLAYERS)
    state = env.current_state()

    # Initialize agents to be trained
    agents = [ReinforcePlayer(env) for i in range(PLAYERS)]

    # Train all agents
    for i in range(EPISODES):
        run_one_episode(state, agents)
        print('Episode ' + str(i) + ': Player ', env.winning_player + 1, ' wins')

    # Initialize agents for test
    win = 0
    agents[0].epsilon = 0
    for i in range(1, PLAYERS):
        agents[i] = RandomPlayer()

    # Test first agent vs randoms
    for i in range(EPISODES):
        run_one_episode(state, agents)
        if env.winning_player == 1:
            win += 1
        # for i in range(5):
        #    loop(agents)
    print('winrate ', win / EPISODES)
