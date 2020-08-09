from environment import Ludo
from players.rl_basic_player import RLBasicPlayer
from players.random_player import RandomPlayer
from players.ivan_pesic import IvanPesic
from players.reinforce_player import ReinforcePlayer
from players.ivan_pesic import IvanPesic
from players.ivan_pesic_nebojsa import IvanPesicNebojsa
from players.graph_player import GraphPlayer
from game_loop import loop
import random

TOKENS = 4
TRAIN_EPISODES = 10
TEST_EPISODES = 100
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
        state=nextstate
    if train:
        for i in range(len(agents)):
            agents[i].recalculate_end(env.lose_reward(i))


if __name__ == '__main__':

    # Initialize environment
    env = Ludo(PLAYERS)
    state = env.current_state()

    # Initialize agents to be trained
    agents = [GraphPlayer(env) for i in range(PLAYERS)]

    print(len(agents))
    # Train all agents
    for i in range(TRAIN_EPISODES):
        run_one_episode(state, agents, True)
        print('Episode ' + str(i) + ': Player ', env.winning_player + 1, ' wins')

    # Initialize agents for test
    win = 0
    #agents[0].save("players\saves\RLBasic30000-1.pth")
    #agents[1].save("players\saves\RLBasic30000-2.pth")
    #agents[2].save("players\saves\RLBasic30000-3.pth")
    #agents[3].save("players\saves\RLBasic30000-4.pth")
    #agents[0].epsilon = 0
    for i in range(1, PLAYERS):
        agents[i] = IvanPesic(env)

    #agents[0].save("src\players\saves\RlBasic.pth")

    # Test first agent vs randoms
    for i in range(TEST_EPISODES):
        run_one_episode(state, agents, False)
        #print(env.winning_player)
        if env.winning_player == 0:
            win += 1
    #loop(agents)

    print('winrate ', win / TEST_EPISODES)
    # loop(agents)
    # agents[0].save("players\saves\RLBasic1500-alternative.pth")
