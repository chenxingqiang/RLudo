from environment import Ludo
from players.rl_basic_player import RLBasicPlayer
from players.random_player import RandomPlayer
from game_loop import loop
import random

TOKENS = 4
EPISODES = 10
PLAYERS = 4

def iterate(state, agents):
    game_end = False
    env.reset(PLAYERS)
    while not game_end:
        x = env.current_player
        action=agents[x].play(state,TOKENS)
        nextstate, reward, game_end = env.step(action)
        agents[x].recalculate(nextstate, reward)


if __name__ == '__main__':
    env = Ludo(PLAYERS)
    if PLAYERS==2:
        agents = [RLBasicPlayer(env) for i in range(2)]
        state = env.current_state()
        for i in range(EPISODES):
            iterate(state, agents)
            print('Episode ' + str(i) + ': Player ', env.winning_player + 1, ' wins')
        win = 0
        agents[0].epsilon=0
        agents[1] = RandomPlayer()
        for i in range(EPISODES):
            iterate(state, agents)
            if env.winning_player == 1:
                win += 1
        print('winrate ', win / EPISODES)
    if PLAYERS==4:
        agents = [RLBasicPlayer(env) for i in range(4)]
        state = env.current_state()
        for i in range(EPISODES):
            iterate(state, agents)
            print('Episode ' + str(i) + ': Player ', env.winning_player + 1, ' wins')
        win = 0
        agents[0].epsilon=0
        agents[1]=RandomPlayer()
        agents[2]=RandomPlayer()
        agents[3]=RandomPlayer()
        for i in range(EPISODES):
            iterate(state, agents)
            if env.winning_player == 1:
                win += 1
        for i in range(5): loop(agents)
        print('winrate ', win / EPISODES)
