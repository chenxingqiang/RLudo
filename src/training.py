from environment import Ludo
from players.rl_basic_player import RLBasicPlayer
from players.random_player import RandomPlayer
import random
TOKENS = 4
EPISODES = 10000
if __name__ == '__main__':
    env=Ludo(2)
    game_end=False
    agent1=RLBasicPlayer(env)
    agent2=RLBasicPlayer(env)
    state=env.current_state
    for i in range(EPISODES):
        env.reset(2)
        while not game_end:
            roll=random.randrange(1,7)
            x=env.current_player
            if(x==0): action=agent1.play(state,TOKENS)
            if(x==1): action=agent2.play(state,TOKENS)
            nextstate,reward,game_end=env.step(roll,action)
            if(x==0): agent1.recalculate(nextstate, reward)
            if(x==1): agent2.recalculate(nextstate, reward)
        print('Player ',env.winning_player+1,' wins')
    win=0
    agentr=RandomPlayer()
    for i in range(EPISODES):
        env.reset(2)
        while not game_end:
            roll=random.randrange(1,7)
            x=env.current_player
            if(x==0): action=agent1.play(state,TOKENS)
            if(x==1): action=agentr.play(state,TOKENS)
            nextstate,reward,game_end=env.step(roll,action)
            if(x==0): agent1.recalculate(nextstate, reward)
            if(x==1): agentr.recalculate(nextstate, reward)
        if env.winning_player==1 win+=1
    print('winrate ',win/EPISODES)
