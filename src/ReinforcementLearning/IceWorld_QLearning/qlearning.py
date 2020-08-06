import random

from ice_env import *
import torch

EPISODES = 1000
EPSILON = 0.15 #Exploration vs Exploitation
GAMMA = 0.9
LEARNING_RATE = 0.1


def main():
    env = Ice()
    average_cumulative_reward = 0.0

    # Q-table, for each env state have action space
    # (current example: 4x4 states, 4 actions per state)
    qtable = torch.zeros(env.env_space() + env.action_space(), dtype=torch.float)

    # Loop over episodes
    for i in range(EPISODES):
        state = env.reset()
        terminate = False
        cumulative_reward = 0.0
        
        # Loop over time-steps
        while not terminate:
            qvalue = qtable[state]
            greedy = torch.argmax(qvalue)
            if (random.random() < EPSILON):
                action = random.randrange(env.action_space()[0])
            else:
                action = greedy

                # 1.2 Sometimes, the agent takes a random action, to explore the environment

            # 2 Perform the action
            nextstate, reward, terminate = env.step(action)

            # 3 Update the q-table
            td_error = reward + max(qtable[nextstate]) * GAMMA - qtable[state][action]
            qtable[state][action] += td_error * LEARNING_RATE

            # 4 Update cumulative reward
            cumulative_reward += reward

            # 5 Make current state next state
            state = nextstate

            continue

        print(i, cumulative_reward)

    env.print_qtable_stats(qtable)

if __name__ == '__main__':
    main()