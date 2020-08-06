from environment import Ludo
from players.rl_basic_player import RLBasicPlayer
from players.random_player import RandomPlayer
import random

TOKENS = 4
EPISODES = 20


def iterate(state, agent1, agent2):
    game_end = False
    env.reset(2)
    while not game_end:
        x = env.current_player
        if x == 0:
            action = agent1.play(state, TOKENS)
        if x == 1:
            action = agent2.play(state, TOKENS)
        nextstate, reward, game_end = env.step(action)
        if x == 0:
            agent1.recalculate(nextstate, reward)
        if x == 1:
            agent2.recalculate(nextstate, reward)


if __name__ == '__main__':
    env = Ludo(2)
    agent1 = RLBasicPlayer(env)
    agent2 = RLBasicPlayer(env)
    state = env.current_state()
    for i in range(EPISODES):
        iterate(state, agent1, agent2)
        print('Episode ' + str(i) + ': Player ', env.winning_player + 1, ' wins')
    win = 0
    agentr = RandomPlayer()
    for i in range(EPISODES):
        iterate(state, agent1, agentr)
        if env.winning_player == 1:
            win += 1
    print('winrate ', win / EPISODES)
