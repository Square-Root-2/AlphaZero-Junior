from AlphaZero import AlphaZero
from BitboardType import BitboardType
from Game import Game
import MoveIndexing
from MoveGenerator import MoveGenerator
from Node import Node
from State import State

import time
from typing import List


def a_to_lan(a, state: State):
    starting_square_index = a // MoveIndexing.MOVES_PER_LOCATION
    move_index = a % MoveIndexing.MOVES_PER_LOCATION

    i = starting_square_index // 8
    j = starting_square_index % 8
    promotion_character = ''

    if move_index < len(MoveGenerator.Direction) * MoveIndexing.MOVES_PER_DIRECTION:  # directional moves
        di = [-1, -1, 0, 1, 1, 1, 0, -1]
        dj = [0, 1, 1, 1, 0, -1, -1, -1]

        direction_index = move_index // MoveIndexing.MOVES_PER_DIRECTION
        magnitude_index = move_index % MoveIndexing.MOVES_PER_DIRECTION + 1
        k = i + magnitude_index * di[direction_index]
        l = j + magnitude_index * dj[direction_index]

        # promotion to queen
        if state.active_color:
            promotion_i = 7
        else:
            promotion_i = 0

        if state.mailbox[8 * i + j].lower() == 'p' and k == promotion_i:
            promotion_character = 'q'
    elif move_index < MoveIndexing.KNIGHT_MOVE_BEGINNING_INDEX + MoveIndexing.KNIGHT_MOVES:  # knight moves
        di = [-2, -1, 1, 2, 2, 1, -1, -2]
        dj = [1, 2, 2, 1, -1, -2, -2, -1]

        direction_index = move_index - MoveIndexing.KNIGHT_MOVE_BEGINNING_INDEX
        k = i + di[direction_index]
        l = j + dj[direction_index]
    else:  # underpromotions
        if state.active_color:
            k = 7
        else:
            k = 0

        dj = [0, -1, 1]
        piece = ['r', 'b', 'n']

        direction_index = (move_index - MoveIndexing.UNDERPROMOTION_BEGINNING_INDEX) // 3
        piece_index = (move_index - MoveIndexing.UNDERPROMOTION_BEGINNING_INDEX) % 3
        l = j + dj[direction_index]
        promotion_character = piece[piece_index]

    return chr(ord('a') + j) + str(8 - i) + chr(ord('a') + l) + str(8 - k) + promotion_character


# Returns True if and only if the following conditions are met:
# 1. The LAN is between 4 to 5 characters long.
# 2. The first and third characters of the LAN are letters between 'a' and 'h', inclusive.
# 3. The second and fourth characters of the LAN are digits between 1 and 8, inclusive.
# 4. The fifth character, if it exists, is 'n', 'b', 'r', or 'q'.
def is_lan_legal(lan):
    if len(lan) < 4 or len(lan) > 5:
        return False

    if ord(lan[0].lower()) < ord('a') or ord(lan[0].lower()) > ord('h'):
        return False

    if ord(lan[1]) < ord('1') or ord(lan[1]) > ord('8'):
        return False

    if ord(lan[2].lower()) < ord('a') or ord(lan[2].lower()) > ord('h'):
        return False

    if ord(lan[3]) < ord('1') or ord(lan[3]) > ord('8'):
        return False

    if len(lan) == 5 and lan[4] not in ['n', 'b', 'r', 'q']:
        return False

    return True


def is_timeout_vs_insufficient_material(game: Game):
    if game.to_play():
        queen_bitboard = game.history[-1].bitboards[BitboardType.WHITE_QUEEN]
        rook_bitboard = game.history[-1].bitboards[BitboardType.WHITE_ROOK]
        bishop_bitboard = game.history[-1].bitboards[BitboardType.WHITE_BISHOP]
        knight_bitboard = game.history[-1].bitobards[BitboardType.WHITE_KNIGHT]
    else:
        queen_bitboard = game.history[-1].bitboards[BitboardType.BLACK_QUEEN]
        rook_bitboard = game.history[-1].bitboards[BitboardType.BLACK_ROOK]
        bishop_bitboard = game.history[-1].bitboards[BitboardType.BLACK_BISHOP]
        knight_bitboard = game.history[-1].bitobards[BitboardType.BLACK_KNIGHT]

    if queen_bitboard > 0:
        return False

    if rook_bitboard > 0:
        return False

    if bishop_bitboard.bit_count() >= 2:
        return False

    if bishop_bitboard > 0 and knight_bitboard > 0:
        return False

    return True


def lan_to_a(lan):
    if lan.lower() == 'resign':
        return MoveIndexing.TOTAL_MOVES

    if not is_lan_legal(lan):
        return -1

    i = 8 - int(lan[1])
    j = ord(lan[0].lower()) - ord('a')
    k = 8 - int(lan[3])
    l = ord(lan[2].lower()) - ord('a')

    starting_square_index = 8 * i + j

    if len(lan) == 5 and lan[4].lower() in ['r', 'b', 'n']:  # underpromotions
        move_index = MoveIndexing.UNDERPROMOTION_BEGINNING_INDEX

        if i == 1:
            target_di = -1
        else:
            target_di = 1

        dj_to_underpromotion_direction_index = {0: 0, -1: 1, 1: 2}
        character_to_underpromotion_index = {'r': 0, 'b': 1, 'n': 2}

        if l - j in dj_to_underpromotion_direction_index.keys() and k - i == target_di:
            move_index += dj_to_underpromotion_direction_index[l - j] + character_to_underpromotion_index[
                lan[4].lower()]
        else:
            move_index = -1
    else:  # directional and knight moves
        move_index = max(MoveGenerator.get_directional_move_index(k - i, l - j),
                         MoveGenerator.get_knight_move_index(k - i, l - j))

    if move_index == -1:
        return -1

    return MoveIndexing.MOVES_PER_LOCATION * starting_square_index + move_index


def print_board(player, timer: List[float], mailbox: str):
    if player:
        print("Computer: " + str(timer[0]) + "s", end='\n\n')

        print("  h g f e d c b a")
        for i in range(7, -1, -1):
            print(8 - i, end=' ')
            for j in range(7, -1, -1):
                print(mailbox[8 * i + j], end=' ')
            print()

        print()

        print("Player: " + str(timer[1]) + "s", end='\n\n')

    else:
        print("Computer: " + str(timer[1]) + "s", end='\n\n')

        print("  a b c d e f g h")
        for i in range(8):
            print(8 - i, end=' ')
            for j in range(8):
                print(mailbox[8 * i + j], end=' ')
            print()
        print()

        print("Player: " + str(timer[0]) + "s", end='\n\n')


MoveGenerator.initialize()
alpha_zero = AlphaZero()
while True:
    game = Game()

    print("To input a move, enter the coordinates of the start square, followed by the coordinates of the end\n"
          "square, without spaces (e.g. e2e4). (For castling, the start and end squares are determined by the\n"
          "king (e.g. e1g1).) For promotions, follow the coordinates of the start and end squares with a\n"
          "character representing the promoted-to piece (n = knight, b = bishop, r = rook, q = queen) (e.g.\n"
          "e7e8q). To resign, enter \"resign\".", end='\n\n')

    while True:
        player = input("Color (0 = white, 1 = black): ")
        if player in ['0', '1']:
            break
    player = bool(int(player))

    minutes = 10
    while True:
        try:
            minutes = float(input("Time per player (in minutes): "))
            break
        except ValueError:
            continue

    increment = 0
    while True:
        try:
            increment = float(input("Increment per player (in seconds): "))
            break
        except ValueError:
            continue

    print()

    timer = 2 * [60 * minutes]

    print_board(player, timer, game.history[-1].mailbox)

    while not game.terminal():
        if game.to_play() == player:
            start = time.time()
            while True:
                action = lan_to_a(input("Move: "))
                if action in game.legal_actions() or action == MoveIndexing.TOTAL_MOVES or time.time() - start > timer[player]:
                    break

            print()

            timer[player] -= time.time() - start

            if timer[player] < 0:
                if is_timeout_vs_insufficient_material(game):
                    print("Draw by timeout vs. insufficient material.", end='\n\n')
                else:
                    print("Player lost on time.", end='\n\n')
                break

            if action == MoveIndexing.TOTAL_MOVES:
                print("Player lost by resignation.", end='\n\n')
                break

            timer[player] += increment
        else:
            print("Computer is thinking...")

            start = time.time()
            root = Node(0)
            action = alpha_zero.run_mcts(game, root, timer[not player] / 20)
            timer[not player] -= time.time() - start

            print("Move: " + a_to_lan(action, game.history[-1]), end='\n\n')

            if timer[not player] < 0:
                if is_timeout_vs_insufficient_material(game):
                    print("Draw by timeout vs. insufficient material.", end='\n\n')
                else:
                    print("Computer lost on time.", end='\n\n')
                break

            timer[not player] += increment

        game.apply(action)
        print_board(player, timer, game.history[-1].mailbox)

    if game.terminal():
        if game.terminal_value(game.to_play()) == 0:
            print("Draw by " + game.terminal_string + ".", end='\n\n')
        elif game.to_play() == player:
            print("Computer wins by " + game.terminal_string + ".", end='\n\n')
        elif game.to_play() == (not player):
            print("Player wins by " + game.terminal_string + ".", end='\n\n')

    while True:
        again = input("Play again (Y = yes, N = no)?: ")
        if again.lower() in ['y', 'n']:
            break

    if again.lower() == 'n':
        break

    print('\n')
