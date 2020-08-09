import curses
import curses.textpad
import random
from environment import Ludo
from players.random_player import RandomPlayer
from players.rl_basic_player import RLBasicPlayer
from players.ivan_pesic import IvanPesic
from players.reinforce_player import ReinforcePlayer
#from players.ivan_pesic_nebojsa import IvanPesicNebojsa
from players.human_player import HumanPlayer
from players.graph_player import GraphPlayer

# Radi samo iz komandne linije

PATH_CHAR = 'X'
HOME_CHAR = 'O'
PIECE_CHAR = 'P'
TOKENS = 4
PLAYERS = 4
matrix = [[-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
          [-1, -1, -1, -1, -1, 18, 19, 20, -1, -1, -1, -1, -1],
          [-1, -1, -1, -1, -1, 17, -1, 21, -1, -1, -1, -1, -1],
          [-1, -1, -1, -1, -1, 16, -1, 22, -1, -1, -1, -1, -1],
          [-1, -1, -1, -1, -1, 15, -1, 23, -1, -1, -1, -1, -1],
          [-1, 10, 11, 12, 13, 14, -1, 24, 25, 26, 27, 28, -1],
          [-1, 9, -1, -1, -1, -1, -1, -1, -1, -1, -1, 29, -1],
          [-1, 8, 7, 6, 5, 4, -1, 34, 33, 32, 31, 30, -1],
          [-1, -1, -1, -1, -1, 3, -1, 35, -1, -1, -1, -1, -1],
          [-1, -1, -1, -1, -1, 2, -1, 36, -1, -1, -1, -1, -1],
          [-1, -1, -1, -1, -1, 1, -1, 37, -1, -1, -1, -1, -1],
          [-1, -1, -1, -1, -1, 0, 39, 38, -1, -1, -1, -1, -1],
          [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]]


def init_colors():
    curses.start_color()
    curses.init_pair(1, curses.COLOR_RED, curses.COLOR_WHITE)  # PLAYER 1
    curses.init_pair(2, curses.COLOR_GREEN, curses.COLOR_WHITE)  # PLAYER 2
    curses.init_pair(3, curses.COLOR_BLUE, curses.COLOR_WHITE)  # PLAYER 3
    curses.init_pair(4, curses.COLOR_YELLOW, curses.COLOR_WHITE)  # PLAYER 4
    curses.init_pair(255, curses.COLOR_BLACK, curses.COLOR_WHITE)  # PATH
    return


def empty_board(dim):
    window = curses.newwin(2 * dim + 5, 2 * dim + 5)

    for i in range(dim + 1, dim + 4):
        for j in range(1, 2 * dim + 4):
            window.addch(i, j, PATH_CHAR, curses.color_pair(255))
            window.addch(j, i, PATH_CHAR, curses.color_pair(255))
    for i in range(dim + 2, 2 * dim + 3):  # 1
        window.addch(i, dim + 2, HOME_CHAR, curses.color_pair(1))
    for i in range(2, dim + 2):  # 2
        window.addch(dim + 2, i, HOME_CHAR, curses.color_pair(3))
    for i in range(2, dim + 2):  # 3
        window.addch(i, dim + 2, HOME_CHAR, curses.color_pair(2))
    for i in range(dim + 2, 2 * dim + 3):  # 4
        window.addch(dim + 2, i, HOME_CHAR, curses.color_pair(4))
    window.addch(dim + 2, dim + 2, ' ')
    return window


def draw_board(dim, state, env):
    window = curses.newwin(2 * dim + 8, 2 * dim + 30)

    for i in range(dim + 1, dim + 4):
        for j in range(1, 2 * dim + 4):
            if state[0][matrix[i][j]] == 0:
                window.addch(i, j, PATH_CHAR, curses.color_pair(255))
            else:
                window.addch(i, j, PIECE_CHAR, curses.color_pair((state[0][matrix[i][j]] - 1) // TOKENS + 1))
            if state[0][matrix[j][i]] == 0:
                window.addch(j, i, PATH_CHAR, curses.color_pair(255))
            else:
                window.addch(j, i, PIECE_CHAR, curses.color_pair((state[0][matrix[j][i]] - 1) // TOKENS + 1))
    for i in range(dim + 3, 2 * dim + 3):  # 1
        if state[1][0][i - dim - 3] == 0:
            window.addch(i, dim + 2, HOME_CHAR, curses.color_pair(1))
        else:
            window.addch(i, dim + 2, PIECE_CHAR, curses.color_pair(1))
    for i in range(2, dim + 2):  # 3
        if state[1][2][i - 2] == 0:
            window.addch(dim + 2, i, HOME_CHAR, curses.color_pair(3))
        else:
            window.addch(dim + 2, i, PIECE_CHAR, curses.color_pair(3))
    for i in range(2, dim + 2):  # 2
        if state[1][1][i - 2] == 0:
            window.addch(i, dim + 2, HOME_CHAR, curses.color_pair(2))
        else:
            window.addch(i, dim + 2, PIECE_CHAR, curses.color_pair(2))
    for i in range(dim + 2, 2 * dim + 3):  # 4
        if state[1][3][i - dim - 3] == 0:
            window.addch(dim + 2, i, HOME_CHAR, curses.color_pair(4))
        else:
            window.addch(dim + 2, i, PIECE_CHAR, curses.color_pair(4))
    window.addch(dim + 2, dim + 2, ' ')
    if env.positions[0][0] == -1 and state[1][0][0] == 0:
        window.addch(2 * dim + 3, 1, PIECE_CHAR, curses.color_pair(1))
    else:
        window.addch(2 * dim + 3, 1, PATH_CHAR, curses.color_pair(1))
    if env.positions[0][1] == -1 and state[1][0][1] == 0:
        window.addch(2 * dim + 3, 2, PIECE_CHAR, curses.color_pair(1))
    else:
        window.addch(2 * dim + 3, 2, PATH_CHAR, curses.color_pair(1))
    if env.positions[0][2] == -1 and state[1][0][2] == 0:
        window.addch(2 * dim + 2, 1, PIECE_CHAR, curses.color_pair(1))
    else:
        window.addch(2 * dim + 2, 1, PATH_CHAR, curses.color_pair(1))
    if env.positions[0][3] == -1 and state[1][0][3] == 0:
        window.addch(2 * dim + 2, 2, PIECE_CHAR, curses.color_pair(1))
    else:
        window.addch(2 * dim + 2, 2, PATH_CHAR, curses.color_pair(1))

    if env.positions[1][0] == -1 and state[1][1][0] == 0:
        window.addch(1, 2 * dim + 3, PIECE_CHAR, curses.color_pair(2))
    else:
        window.addch(1, 2 * dim + 3, PATH_CHAR, curses.color_pair(2))

    if env.positions[1][1] == -1 and state[1][1][1] == 0:
        window.addch(2, 2 * dim + 3, PIECE_CHAR, curses.color_pair(2))
    else:
        window.addch(2, 2 * dim + 3, PATH_CHAR, curses.color_pair(2))
    if env.positions[1][2] == -1 and state[1][1][2] == 0:
        window.addch(1, 2 * dim + 2, PIECE_CHAR, curses.color_pair(2))
    else:
        window.addch(1, 2 * dim + 2, PATH_CHAR, curses.color_pair(2))

    if env.positions[1][3] == -1 and state[1][1][3] == 0:
        window.addch(2, 2 * dim + 2, PIECE_CHAR, curses.color_pair(2))
    else:
        window.addch(2, 2 * dim + 2, PATH_CHAR, curses.color_pair(2))

    if env.positions[2][0] == -1 and state[1][2][0] == 0:
        window.addch(1, 1, PIECE_CHAR, curses.color_pair(3))
    else:
        window.addch(1, 1, PATH_CHAR, curses.color_pair(3))

    if env.positions[2][1] == -1 and state[1][2][1] == 0:
        window.addch(1, 2, PIECE_CHAR, curses.color_pair(3))
    else:
        window.addch(1, 2, PATH_CHAR, curses.color_pair(3))
    if env.positions[2][2] == -1 and state[1][2][2] == 0:
        window.addch(2, 1, PIECE_CHAR, curses.color_pair(3))
    else:
        window.addch(2, 1, PATH_CHAR, curses.color_pair(3))

    if env.positions[2][3] == -1 and state[1][2][3] == 0:
        window.addch(2, 2, PIECE_CHAR, curses.color_pair(3))
    else:
        window.addch(2, 2, PATH_CHAR, curses.color_pair(3))

    if env.positions[3][0] == -1 and state[1][3][0] == 0:
        window.addch(2 * dim + 3, 2 * dim + 3, PIECE_CHAR, curses.color_pair(4))
    else:
        window.addch(2 * dim + 3, 2 * dim + 3, PATH_CHAR, curses.color_pair(4))

    if env.positions[3][1] == -1 and state[1][3][1] == 0:
        window.addch(2 * dim + 3, 2 * dim + 2, PIECE_CHAR, curses.color_pair(4))
    else:
        window.addch(2 * dim + 3, 2 * dim + 2, PATH_CHAR, curses.color_pair(4))
    if env.positions[3][2] == -1 and state[1][3][2] == 0:
        window.addch(2 * dim + 2, 2 * dim + 3, PIECE_CHAR, curses.color_pair(4))
    else:
        window.addch(2 * dim + 2, 2 * dim + 3, PATH_CHAR, curses.color_pair(4))

    if env.positions[3][3] == -1 and state[1][3][3] == 0:
        window.addch(2 * dim + 2, 2 * dim + 2, PIECE_CHAR, curses.color_pair(4))
    else:
        window.addch(2 * dim + 2, 2 * dim + 2, PATH_CHAR, curses.color_pair(4))
    return window

def human_board(dim, state, env,player):
    window = curses.newwin(2 * dim + 8, 2 * dim + 30)
    PIECE=[['a' for j in range(TOKENS)] for i in range(PLAYERS)]
    for i in range(PLAYERS):
        for j in range(TOKENS):
            if i!=player:
                PIECE[i][j]=PIECE_CHAR
            else:
                PIECE[i][j]=chr(ord('1')+j)
    for i in range(dim + 1, dim + 4):
        for j in range(1, 2 * dim + 4):
            plr=(state[0][matrix[i][j]] - 1) // TOKENS
            tkn=(state[0][matrix[i][j]] - 1) % TOKENS
            if state[0][matrix[i][j]] == 0:
                window.addch(i, j, PATH_CHAR, curses.color_pair(255))
            else:
                window.addch(i, j, PIECE[plr][tkn], curses.color_pair(plr+1))
            plr = (state[0][matrix[j][i]] - 1) // TOKENS
            tkn = (state[0][matrix[j][i]] - 1) % TOKENS
            if state[0][matrix[j][i]] == 0:
                window.addch(j, i, PATH_CHAR, curses.color_pair(255))
            else:
                window.addch(j, i, PIECE[plr][tkn], curses.color_pair(plr+1))
    for i in range(dim + 3, 2 * dim + 3):  # 1
        if state[1][0][i - dim - 3] == 0:
            window.addch(i, dim + 2, HOME_CHAR, curses.color_pair(1))
        else:
            window.addch(i, dim + 2, PIECE[0][i-dim-3], curses.color_pair(1))
    for i in range(2, dim + 2):  # 3
        if state[1][2][i - 2] == 0:
            window.addch(dim + 2, i, HOME_CHAR, curses.color_pair(3))
        else:
            window.addch(dim + 2, i, PIECE[2][i-2], curses.color_pair(3))
    for i in range(2, dim + 2):  # 2
        if state[1][1][i - 2] == 0:
            window.addch(i, dim + 2, HOME_CHAR, curses.color_pair(2))
        else:
            window.addch(i, dim + 2, PIECE[1][i-2], curses.color_pair(2))
    for i in range(dim + 2, 2 * dim + 3):  # 4
        if state[1][3][i - dim - 3] == 0:
            window.addch(dim + 2, i, HOME_CHAR, curses.color_pair(4))
        else:
            window.addch(dim + 2, i, PIECE[3][i-dim-3], curses.color_pair(4))
    window.addch(dim + 2, dim + 2, ' ')
    if env.positions[0][0] == -1 and state[1][0][0] == 0:
        window.addch(2 * dim + 3, 1, PIECE[0][0], curses.color_pair(1))
    else:
        window.addch(2 * dim + 3, 1, PATH_CHAR, curses.color_pair(1))
    if env.positions[0][1] == -1 and state[1][0][1] == 0:
        window.addch(2 * dim + 3, 2, PIECE[0][1], curses.color_pair(1))
    else:
        window.addch(2 * dim + 3, 2, PATH_CHAR, curses.color_pair(1))
    if env.positions[0][2] == -1 and state[1][0][2] == 0:
        window.addch(2 * dim + 2, 1, PIECE[0][2], curses.color_pair(1))
    else:
        window.addch(2 * dim + 2, 1, PATH_CHAR, curses.color_pair(1))
    if env.positions[0][3] == -1 and state[1][0][3] == 0:
        window.addch(2 * dim + 2, 2, PIECE[0][3], curses.color_pair(1))
    else:
        window.addch(2 * dim + 2, 2, PATH_CHAR, curses.color_pair(1))

    if env.positions[1][0] == -1 and state[1][1][0] == 0:
        window.addch(1, 2 * dim + 3, PIECE[1][0], curses.color_pair(2))
    else:
        window.addch(1, 2 * dim + 3, PATH_CHAR, curses.color_pair(2))

    if env.positions[1][1] == -1 and state[1][1][1] == 0:
        window.addch(2, 2 * dim + 3, PIECE[1][1], curses.color_pair(2))
    else:
        window.addch(2, 2 * dim + 3, PATH_CHAR, curses.color_pair(2))
    if env.positions[1][2] == -1 and state[1][1][2] == 0:
        window.addch(1, 2 * dim + 2, PIECE[1][2], curses.color_pair(2))
    else:
        window.addch(1, 2 * dim + 2, PATH_CHAR, curses.color_pair(2))

    if env.positions[1][3] == -1 and state[1][1][3] == 0:
        window.addch(2, 2 * dim + 2, PIECE[1][3], curses.color_pair(2))
    else:
        window.addch(2, 2 * dim + 2, PATH_CHAR, curses.color_pair(2))

    if env.positions[2][0] == -1 and state[1][2][0] == 0:
        window.addch(1, 1, PIECE[2][0], curses.color_pair(3))
    else:
        window.addch(1, 1, PATH_CHAR, curses.color_pair(3))

    if env.positions[2][1] == -1 and state[1][2][1] == 0:
        window.addch(1, 2, PIECE[2][1], curses.color_pair(3))
    else:
        window.addch(1, 2, PATH_CHAR, curses.color_pair(3))
    if env.positions[2][2] == -1 and state[1][2][2] == 0:
        window.addch(2, 1, PIECE[2][2], curses.color_pair(3))
    else:
        window.addch(2, 1, PATH_CHAR, curses.color_pair(3))

    if env.positions[2][3] == -1 and state[1][2][3] == 0:
        window.addch(2, 2, PIECE[2][3], curses.color_pair(3))
    else:
        window.addch(2, 2, PATH_CHAR, curses.color_pair(3))

    if env.positions[3][0] == -1 and state[1][3][0] == 0:
        window.addch(2 * dim + 3, 2 * dim + 3, PIECE[3][0], curses.color_pair(4))
    else:
        window.addch(2 * dim + 3, 2 * dim + 3, PATH_CHAR, curses.color_pair(4))

    if env.positions[3][1] == -1 and state[1][3][1] == 0:
        window.addch(2 * dim + 3, 2 * dim + 2, PIECE[3][1], curses.color_pair(4))
    else:
        window.addch(2 * dim + 3, 2 * dim + 2, PATH_CHAR, curses.color_pair(4))
    if env.positions[3][2] == -1 and state[1][3][2] == 0:
        window.addch(2 * dim + 2, 2 * dim + 3, PIECE[3][2], curses.color_pair(4))
    else:
        window.addch(2 * dim + 2, 2 * dim + 3, PATH_CHAR, curses.color_pair(4))

    if env.positions[3][3] == -1 and state[1][3][3] == 0:
        window.addch(2 * dim + 2, 2 * dim + 2, PIECE[3][3], curses.color_pair(4))
    else:
        window.addch(2 * dim + 2, 2 * dim + 2, PATH_CHAR, curses.color_pair(4))
    return window


def raw_loop(screen):
    screen.clear()
    curses.curs_set(0)
    init_colors()
    empty_board(4).refresh()
    game_end = False
    env = Ludo(PLAYERS)
    global agents
    if agents==None:
        agents = [IvanPesic(env) for i in range(PLAYERS)]
    agents[0]=ReinforcePlayer(env, "players\saves\Reinforce30000-1.pth")
    agents[1]=RLBasicPlayer(env, "players\saves\RLBasic30000-2.pth")
    agents[3]=HumanPlayer(env)
    if agents == None:
        agents = [RandomPlayer() for i in range(PLAYERS)]
    pstate = env.current_state()
    while not game_end:
        if isinstance(agents[env.current_player],HumanPlayer):    
            window=human_board(4, state, env,env.current_player)
            window.addstr(2*4+5,0,'Igrac ')
            window.addstr(2*4+5,6,str(env.current_player+1))
            window.addstr(2*4+5,8,'je na potezu')
            window.addstr(2*4+6,0,'Na kocki je bacen broj ')
            window.addstr(2*4+6,23,str(env.roll+1))
            while True:
                window.refresh()
                curses.napms(30)
                c = window.getch()
                action=agents[env.current_player].play(c)
                if not action == -1:
                    break
                window.addstr(2*4+7,0,'Morate odigrati validan potez')
        else:
            action = agents[env.current_player].play(pstate, TOKENS)
        pstate, r, game_end = env.step(action)
        state = env.current_state_as_tuple()
        draw_board(4, state, env).refresh()
        curses.napms(30)
    curses.curs_set(1)
    print('Player ', env.winning_player + 1, ' wins')


def loop(ag=None):
    global agents
    agents = ag
    curses.wrapper(raw_loop)
    # raw_loop(None)
    return
