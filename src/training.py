from environment import Ludo
from players.rl_basic_player import RLBasicPlayer
from players.random_player import RandomPlayer
from players.ivan_pesic import IvanPesic
from players.reinforce_player import ReinforcePlayer
from players.ivan_pesic import IvanPesic
from game_loop import loop
import random

TOKENS = 4
TRAIN_EPISODES = 0
TEST_EPISODES = 20
PLAYERS = 4


def run_one_episode(state, agents, train):
    game_end = False
    env.reset(PLAYERS)
    for agent in agents:
        agent.reset()
    while not game_end:
        x = env.current_player
        action = agents[x].play(state, TOKENS)
        nextstate, reward, game_end = env.step(action)
        if train:
            agents[x].recalculate_step(nextstate, reward)
        state = nextstate
    if train:
        for i in range(len(agents)):
            agents[i].recalculate_end(env.lose_reward(i))


if __name__ == '__main__':

    # Initialize environment
    env = Ludo(PLAYERS)
    state = env.current_state()

    # Initialize agents to be trained
    agents = [RLBasicPlayer(env) for i in range(PLAYERS)]


    # Train all agents
    for i in range(TRAIN_EPISODES):
        run_one_episode(state, agents, True)
        print('Episode ' + str(i) + ': Player ', env.winning_player + 1, ' wins')

    # Initialize agents for test
    win = 0
    agents[0].epsilon = 0
    for i in range(1, PLAYERS):
        agents[i] = IvanPesic(env)

    # Test first agent vs randoms
    for i in range(TEST_EPISODES):
        run_one_episode(state, agents, False)
        if env.winning_player == 1:
            win += 1
    #loop(agents)
    agents[0].save("players\saves\RLBasic10.pth")

    print('winrate ', win / TEST_EPISODES)
    #loop(agents)
    #agents[0].save("players\saves\RLBasic1500.pth")

