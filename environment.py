import random
import torch

START_DISTANCE = 10
MAX_PLAYERS = 4
TOKENS_PER_PLAYER = 4

BOARD_LENGTH = MAX_PLAYERS * START_DISTANCE



#ACTIONS = ['^', 'v', '<', '>']
#REWARD = {' ': -5.0, '*': -10.0, 'T': 20.0, 'G': 100.0}
#TERMINAL = {' ': False, '*': True, 'T': False, 'G': True}

class Ludo(object):
    def __init__(self,num_players):
        self.reset(num_players)

    def reset(self,num_players):
        """ Reset the environment and return the initial state number
        """
        self.num_players=num_players
        self.positions=torch.ones(size=(num_players,TOKENS_PER_PLAYER),dtype=int)*(-1)
        self.board_state = torch.zeros(BOARD_LENGTH,dtype=int)
        self.starts=torch.zeros(num_players, dtype=int)
        if self.num_players==2:
            self.starts[0]=0
            self.starts[1]=2*START_DISTANCE
        if self.num_players==3:
            self.starts[0]=0
            self.starts[1]=2*START_DISTANCE
            self.starts[2]=1*START_DISTANCE
        if self.num_players==4:
            self.starts[0]=0
            self.starts[1]=2*START_DISTANCE
            self.starts[2]=1*START_DISTANCE
            self.starts[3]=3*START_DISTANCE
        self.passed=torch.zeros(size=(self.num_players, TOKENS_PER_PLAYER),dtype=int)
        self.home_state = torch.zeros(size=(self.num_players, TOKENS_PER_PLAYER),dtype=int)
        self.current_player = 0

        return self.current_state()

    def step(self, roll, action):
        terminate=False
        player=self.current_player
        if not roll==6:
            self.current_player+=1
            self.current_player%=self.num_players
        if self.home_state[player,action]:
            return self.current_state(), -100, terminate
        playable=(torch.sum(self.positions[player])==-TOKENS_PER_PLAYER)
        cur=self.positions[player,action]
        if cur==-1:
            if roll==6:
                if not self.board_state[self.starts[player]]==0: 
                    return self.current_state(), -100, terminate
                nxt=self.starts[player]
                delta=1
            elif playable:
               return self.current_state(), -100, terminate
            else: return self.current_state(),0,terminate
        else:
            nxt=self.positions[player,action]+roll
            nxt%=BOARD_LENGTH
            delta=roll
        reward=0
        if self.passed[player,action]+delta>=BOARD_LENGTH:
            self.board_state[self.positions[player,action]]=0
            self.positions[player,action]=-1
            self.home_state[player,action]=1
            if torch.sum(self.home_state[player])==TOKENS_PER_PLAYER:
                terminate=True
                self.winning_player=player
                return self.current_state(),100,terminate
            return self.current_state(),0,terminate
        if (self.board_state[nxt]-1)//TOKENS_PER_PLAYER==player and not self.board_state[nxt]==0:
            return self.current_state(), -100, terminate
        if not self.board_state[nxt]==0:
            self.positions[(self.board_state[nxt]-1)//TOKENS_PER_PLAYER,(self.board_state[nxt]-1)%TOKENS_PER_PLAYER]=-1
            self.passed[(self.board_state[nxt]-1)//TOKENS_PER_PLAYER,(self.board_state[nxt]-1)%TOKENS_PER_PLAYER]=0
        self.passed[player,action]+=delta
        self.board_state[self.positions[player,action]]=0
        self.positions[player,action]=nxt
        self.board_state[nxt]=TOKENS_PER_PLAYER*player+action+1
        return self.current_state(), reward, terminate

    def current_state(self):
        return self.board_state, self.home_state

    def action_space(self):
        return TOKENS_PER_PLAYER,

    def env_space(self):
        #TODO
        return 4, 4

