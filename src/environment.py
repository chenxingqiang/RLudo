import random
import torch
import torch.sparse
import math

START_DISTANCE = 10
MAX_PLAYERS = 4
TOKENS_PER_PLAYER = 4
DICE_MAX = 6

BOARD_LENGTH = MAX_PLAYERS * START_DISTANCE

ILLEGAL_MOVE_REWARD = -2
WIN_REWARD = 20
LOSE_REWARD = -20
CAPTURE_REWARD = 0.1
HOME_RUN_REWARD = 1
FAST_REWARD = 0.2
DEFENSE_REWARD = 0.0005  # To be multiplied with passed
START_REWARD = 0.15

# ACTIONS = ['^', 'v', '<', '>']
# REWARD = {' ': -5.0, '*': -10.0, 'T': 20.0, 'G': 100.0}
# TERMINAL = {' ': False, '*': True, 'T': False, 'G': True}

class Ludo(object):
    def __init__(self, num_players, agents=None):
        self.agents = agents
        self.reset(num_players)
        self.adj = self.sparse_normed_adjacency()

    def reset(self, num_players):
        """
        Reset the environment and return the initial state number
        :param num_players: Number of players
        :return: Initial state
        """

        if torch.cuda.is_available():
            torch.cuda.set_device("cuda:0")

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
        # print(self.board_state,self.roll,self.current_player,action)
        # print(player)
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
            self.current_player=player
            return 0, terminate
        self.running_total = 1

        # Figurica nije jos usla u igru
        if cur == -1:
            # Bacena sestica; figurica ulazi u igru
            if self.roll == DICE_MAX - 1:
                # Ilegalan potez, pomeranje figurice koja je vec na cilju
                if self.home_state[self.current_player][action] == 1:
                    return ILLEGAL_MOVE_REWARD, terminate
                nxt = self.starts[player]
                delta = 1
            # Ilegalan potez, uzaludno pomeranje figurice koja nije u igri, dok postoji drugih poteza
            elif playable:
                return ILLEGAL_MOVE_REWARD, terminate
            else:
                return START_REWARD, terminate
        # pomera figuricu i igra
        else:
            nxt = self.positions[player, action] + self.roll + 1
            nxt %= BOARD_LENGTH
            delta = self.roll + 1

        reward = FAST_REWARD  # Ako pomeramo najdaljeg
        for i in range(TOKENS_PER_PLAYER):
            if self.home_state[player, i] == 1:
                continue
            if self.passed[player, i] > self.passed[player, action]:
                reward = 0
                break
        # Ako branimo
        reward = DEFENSE_REWARD * self.passed[player, action] ** 2 if self.dolazice_po_mene(action) else reward

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
            return HOME_RUN_REWARD, terminate

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
            reward = CAPTURE_REWARD
        self.passed[player, action] += delta
        if not self.positions[player, action]==-1:
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
        # Yeet način da se one-hotuje board state, not recommended
        board_hot = torch.cat([(((self.board_state + 3) // 4) == i + 1).long() for i in range(MAX_PLAYERS)])
        dice_hot = torch.zeros(size=[DICE_MAX], dtype=torch.long)
        dice_hot[self.roll] = 1
        return torch.cat((board_hot,
                          torch.flatten(self.home_state),
                          dice_hot)).float()

    def roll_dice(self):
        """  Dice rolls are handled by environment and are passed with state """
        self.roll = random.randrange(0, DICE_MAX)

    def lose_reward(self, player):
        """
        Returns lose reward if player lost, otherwise None
        :param player: player id
        :return: reward for losing the game
        """
        return None if player == self.winning_player else LOSE_REWARD

    def player_at_pos(self, pos):
        """ Returns player at given position """
        return (self.board_state[(pos + self.roll + 1) % BOARD_LENGTH] - 1) // TOKENS_PER_PLAYER

    ###################
    #    CONSTANTS    #
    ###################

    def dice_max(self):
        return DICE_MAX

    def board_length(self):
        return BOARD_LENGTH

    def players(self):
        return self.num_players

    def action_space(self):
        return TOKENS_PER_PLAYER

    def env_space(self):
        """ Total number of parameters representing a state """
        return MAX_PLAYERS * (TOKENS_PER_PLAYER + BOARD_LENGTH) + DICE_MAX

    ##################################
    #    HERE COME THE HEURISTICS    #
    ##################################

    def is_action_valid(self, action):
        """
        Returns whether given action is a valid move for current player
        :param action: Action to be checked
        :return: Boolean flag
        """
        if self.home_state[self.current_player, action] == 1:
            return False
        pos = self.positions[self.current_player, action]
        if self.positions[self.current_player, action] == -1:
            return self.roll == DICE_MAX - 1
        return self.player_at_pos((pos + self.roll + 1) % BOARD_LENGTH) != self.current_player

    def dolaze_po_mene(self, action):
        """
        DEFENSE HEURISTIC
        Ispitujemo da li je pozicija napadnuta
        """
        my_pos = self.positions[self.current_player, action]
        if my_pos == -1:
            return False
        for pos in range(my_pos - 6, my_pos):
            player_at_pos = self.player_at_pos(action)
            if player_at_pos != -1 and player_at_pos != self.current_player:
                return True
        return False

    def dolazice_po_mene(self, action):
        """
        FUTURE DEFENSE HEURISTIC
        Ispitujemo da li je pozicija napadnuta
        """
        my_pos = self.positions[self.current_player, action]
        if my_pos == -1:
            my_pos = self.starts[self.current_player]
        my_pass = self.passed[self.current_player, action]
        my_pos = my_pass + self.roll + 1
        if my_pos > BOARD_LENGTH:
            return False
        for pos in range(my_pos - 6, my_pos):
            player_at_pos = self.player_at_pos(action)
            if player_at_pos != -1 and player_at_pos != self.current_player:
                return True
        return False

    def ilja_mode(self, action):
        """ ATTACK HEURISTIC """
        my_pos = self.positions[self.current_player, action]
        if my_pos == -1:
            if self.roll != DICE_MAX - 1:
                return False
            x = self.board_state[self.starts[self.current_player]]
            return x != 0 and x != self.current_player + 1
        s = self.board_state[(self.roll + 1 + my_pos) % self.board_length()]
        return s != 0 and s != self.current_player + 1

    #############################
    #    GRAFOVSKE BOLESTINE    #
    #############################

    def to_sparse(self, x):
        """ converts dense tensor x to sparse format """
        x_typename = torch.typename(x).split('.')[-1]
        sparse_tensortype = getattr(torch.sparse, x_typename)

        indices = torch.nonzero(x)
        if len(indices.shape) == 0:  # if all elements are zeros
            return sparse_tensortype(*x.shape)
        indices = indices.t()
        values = x[tuple(indices[i] for i in range(indices.shape[0]))]
        return sparse_tensortype(indices, values, x.size())

    def sparse_normed_adjacency(self):
        """
        Precalculating adjacency matrix for graph nn.
        Ones on the diagonal are known to improve stability.
        :return: Sparse adjacency matrix
        """
        adjmat = torch.zeros(size=(BOARD_LENGTH, BOARD_LENGTH))
        for i in range(BOARD_LENGTH):
            for j in range(BOARD_LENGTH):
                if abs(i - j) <= DICE_MAX:
                    adjmat[i, j] = 1
        # For our needs this can be simplified
        deg = torch.eye(BOARD_LENGTH) / math.sqrt(2*DICE_MAX)
        adjmat = torch.mm(deg, adjmat)
        adjmat = torch.mm(adjmat, deg)
        return self.to_sparse(adjmat)

    def features_matrix(self):
        """
        Returns one-hot encoded board state in matrix form
        :return: matrix of size BOARD_LENGTH * TOKENS_PER_PLAYER
        """
        # TODO: Raditi one-hot bez fora
        feat = torch.zeros(size=(BOARD_LENGTH, TOKENS_PER_PLAYER))
        for i in range(BOARD_LENGTH):
            feat[i, self.player_at_pos(i)] = 1
        return feat