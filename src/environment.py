import random
import torch
import numpy as np

START_DISTANCE = 10
MAX_PLAYERS = 4
TOKENS_PER_PLAYER = 4
DICE_MAX = 6

BOARD_LENGTH = MAX_PLAYERS * START_DISTANCE

ILLEGAL_MOVE_REWARD = -100
WIN_REWARD = 100
LOSE_REWARD = -100


# ACTIONS = ['^', 'v', '<', '>']
# REWARD = {' ': -5.0, '*': -10.0, 'T': 20.0, 'G': 100.0}
# TERMINAL = {' ': False, '*': True, 'T': False, 'G': True}

class Ludo(object):
    def __init__(self, num_players, agents=None):
        self.agents = agents
        self.reset(num_players)

    def reset(self, num_players):
        """
        Reset the environment and return the initial state number
        :param num_players: Number of players
        :return: Initial state
        """

        if torch.cuda.is_available():
            torch.cuda.set_device("cuda:0")
            print("Running on GPU")

        self.ply = 0
        self.num_players = num_players
        self.positions = torch.ones(size=(MAX_PLAYERS, TOKENS_PER_PLAYER), dtype=torch.long) * (-1)
        self.board_state = torch.zeros(BOARD_LENGTH, dtype=torch.long)
        self.starts = torch.zeros(MAX_PLAYERS, dtype=torch.long)
        if self.num_players == 2:
            self.starts[0] = 0
            self.starts[1] = 2 * START_DISTANCE
        if self.num_players == 3:
            self.starts[0] = 0
            self.starts[1] = 2 * START_DISTANCE
            self.starts[2] = 1 * START_DISTANCE
        if self.num_players == 4:
            self.starts[0] = 0
            self.starts[1] = 2 * START_DISTANCE
            self.starts[2] = 1 * START_DISTANCE
            self.starts[3] = 3 * START_DISTANCE
        self.passed = torch.zeros(size=(MAX_PLAYERS, TOKENS_PER_PLAYER), dtype=torch.long)
        self.home_state = torch.zeros(size=(MAX_PLAYERS, TOKENS_PER_PLAYER), dtype=torch.long)
        self.current_player = 0
        self.running_total = 1
        self.roll_dice()

        return self.current_state()

    def move(self, action):
        """
        Execute action given by agent
        :param action: Action to be performed
        :return: Reward and termination flag
        """
        terminate = False
        player = self.current_player
        playable = (torch.sum(self.positions[player]) != -TOKENS_PER_PLAYER)

        print(self.board_state,self.roll,self.current_player,action)

        # ukoliko nije dobio sesticu, menja se na sledeceg igraca
        if not self.roll == DICE_MAX - 1:
            self.current_player += 1
            self.current_player %= self.num_players
        cur = self.positions[player, action]

        # Ilegalan potez, pomeranje figurice koja je vec na cilju
        if self.home_state[player, action]:
            return ILLEGAL_MOVE_REWARD, terminate

        # Bacanje do 3 puta ako nema nista na tabli
        if not playable and self.running_total < 3 and not self.roll == DICE_MAX - 1:
            self.running_total += 1
            return 0, terminate
        self.running_total = 1


        # Figurica nije jos usla u igru
        if cur == -1:
            # Bacena sestica; figurica ulazi u igru
            if self.roll == DICE_MAX - 1:
                # Ilegalan potez, pomeranje figurice koja je vec na cilju
                if not self.board_state[self.starts[player]] == 0:
                    return ILLEGAL_MOVE_REWARD, terminate
                nxt = self.starts[player]
                delta = 1
            # Ilegalan potez, uzaludno pomeranje figurice koja nije u igri, dok postoji drugih poteza
            elif playable:
                return ILLEGAL_MOVE_REWARD, terminate
            else:
                return 0, terminate
        # pomera figuricu i igra
        else:
            nxt = self.positions[player, action] + self.roll
            nxt %= BOARD_LENGTH
            delta = self.roll
        reward = 0

        # Ako je presao celu tablu
        if self.passed[player, action] + delta >= BOARD_LENGTH:
            self.board_state[self.positions[player, action]] = 0
            self.positions[player, action] = -1
            self.home_state[player, action] = 1
            # Sve figurice su stigle na kraj; kraj igre
            if torch.sum(self.home_state[player]) == TOKENS_PER_PLAYER:
                terminate = True
                self.winning_player = player
                return WIN_REWARD, terminate
            self.roll_dice()
            return 0, terminate

        # Ilegalan potez, zavrsio na polju koje je vec zauzeto od strane svoje figure
        if (self.board_state[nxt] - 1) // TOKENS_PER_PLAYER == player \
                and not self.board_state[nxt] == 0:
            return ILLEGAL_MOVE_REWARD, terminate

        # Na polju gde se nalazi vec stoji protivnicka figurica, ona se jede
        if not self.board_state[nxt] == 0:
            self.positions[(self.board_state[nxt] - 1) // TOKENS_PER_PLAYER,
                           (self.board_state[nxt] - 1) % TOKENS_PER_PLAYER] = -1
            self.passed[(self.board_state[nxt] - 1) // TOKENS_PER_PLAYER,
                        (self.board_state[nxt] - 1) % TOKENS_PER_PLAYER] = 0
        self.passed[player, action] += delta
        self.board_state[self.positions[player, action]] = 0
        self.positions[player, action] = nxt
        self.board_state[nxt] = TOKENS_PER_PLAYER * player + action + 1
        return reward, terminate

    def step(self, action):
        """
        Advances environment by one player
        :param action: Action to be performed
        :return: Returns tuple of next state, reward, and termination flag
        """
        # print(self.ply)
        reward, terminate = self.move(action)
        self.ply = self.ply + 1
        self.roll_dice()
        return self.current_state(), reward, terminate

    def current_state_as_tuple(self):
        """
        Returns full current board state as a tuple
        """
        return self.board_state, self.home_state, self.roll

    def current_state(self):
        """
        Returns full current board state as a tensor
        :return: Tensor to be passes as input to nn
        """
        # Yeet način da se one-hotuje board state
        board_hot = torch.cat([(((self.board_state + 3) // 4) == i + 1).long() for i in range(MAX_PLAYERS)])
        dice_hot = torch.zeros(size=[DICE_MAX], dtype=torch.long)
        dice_hot[self.roll] = 1
        return torch.cat((board_hot,
                          torch.flatten(self.home_state),
                          dice_hot)).float()

    def action_space(self):
        return TOKENS_PER_PLAYER

    def env_space(self):
        """
        Total number of parameters representing a state
        :return: Integer
        """
        return MAX_PLAYERS * (TOKENS_PER_PLAYER + BOARD_LENGTH) + DICE_MAX

    def board_length(self):
        return BOARD_LENGTH

    def roll_dice(self):
        """
        Dice rolls are handled by environment and are passed with state
        :return: None
        """
        self.roll = random.randrange(0, DICE_MAX)
        return

    def lose_reward(self, player):
        """
        Returns lose reward if player lost, otherwise None
        :param player: player id
        :return: reward for losing the game
        """
        return None if player == self.winning_player else LOSE_REWARD
