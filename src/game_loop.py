import curses
import curses.textpad
import random
from environment import Ludo
from players.random_player import RandomPlayer

# Radi samo iz komandne linije

PATH_CHAR = 'X'
HOME_CHAR = 'O'
PIECE_CHAR = 'P'
TOKENS = 4
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
    # TODO: Nacrtati početna polja
    return window


def draw_board(dim, state, env):
    window = curses.newwin(2 * dim + 5, 2 * dim + 5)

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


def raw_loop(screen):
    env = Ludo(4)
    screen.clear()
    curses.curs_set(0)
    init_colors()
    empty_board(4).refresh()
    curses.napms(100)
    curses.curs_set(1)
    game_end = False
    agent1 = RandomPlayer()
    agent2 = RandomPlayer()
    agent3 = RandomPlayer()
    agent4 = RandomPlayer()
    state = env.current_state
    while not game_end:
        roll = random.randrange(1, 7)
        if env.current_player == 0:
            action = agent1.play(state, TOKENS)
        elif env.current_player == 1:
            action = agent2.play(state, TOKENS)
        elif env.current_player == 2:
            action = agent3.play(state, TOKENS)
        elif env.current_player == 3:
            action = agent4.play(state, TOKENS)
        else:
            raise RuntimeError()
        state, r, game_end = env.step(roll, action)
        draw_board(4, state, env).refresh()
        curses.napms(30)
    print('Player ', env.winning_player + 1, ' wins')


def loop():
    curses.wrapper(raw_loop)
    return
