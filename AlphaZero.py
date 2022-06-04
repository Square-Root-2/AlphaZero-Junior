from AlphaZeroConfig import AlphaZeroConfig
from BitboardType import BitboardType
from Game import Game
from MoveGenerator import MoveGenerator
import MoveIndexing
from Network import Network
from Node import Node
from State import State
import training

import keras
import time
from typing import List
'''
Returns True if and only if the following conditions are met:
1. The LAN is between 4 to 5 characters long.
2. The first and third characters of the LAN are letters between 'a' and 'h', inclusive.
3. The second and fourth characters of the LAN are digits between 1 and 8, inclusive.
4. The fifth character, if it exists, is 'n', 'b', 'r', or 'q'.
'''


class AlphaZero(object):

    def __init__(self):
        MoveGenerator.initialize()
        self.config = AlphaZeroConfig()
        self.network = Network(keras.models.load_model('model'))

    @staticmethod
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

    @staticmethod
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

    @staticmethod
    def lan_to_move_index(lan):
        if lan.lower() == 'resign':
            return MoveIndexing.TOTAL_MOVES

        if not AlphaZero.is_lan_legal(lan):
            return -1

        i = 8 - int(lan[1])
        j = ord(lan[0].lower()) - ord('a')
        k = 8 - int(lan[3])
        l = ord(lan[2].lower()) - ord('a')

        starting_square_index = 8 * i + j

        # underpromotions
        if len(lan) == 5 and lan[4].lower() in ['r', 'b', 'n']:
            move_index = MoveIndexing.UNDERPROMOTION_BEGINNING_INDEX

            if i == 1:
                target_di = -1
            else:
                target_di = 1

            dj_to_underpromotion_direction_index = {0: 0, -1: 1, 1: 2}
            character_to_underpromotion_index = {'r': 0, 'b': 1, 'n': 2}

            if l - j in dj_to_underpromotion_direction_index.keys() and k - i == target_di:
                move_index += dj_to_underpromotion_direction_index[l - j] + character_to_underpromotion_index[lan[4].lower()]
            else:
                move_index = -1
        # directional and knight moves
        else:
            move_index = max(MoveGenerator.get_directional_move_index(k - i, l - j), MoveGenerator.get_knight_move_index(k - i, l - j))

        if move_index == -1:
            return -1

        return MoveIndexing.MOVES_PER_LOCATION * starting_square_index + move_index

    def play_alpha_zero(self):
        while True:
            game = Game()

            print("To input a move, enter the coordinates of the start square, followed by the coordinates of the end\n"
                  "square, without spaces (e.g. e2e4). (For castling, the start and end squares are determined by the\n"
                  "king (e.g. e1g1).) For promotions, follow the coordinates of the start and end squares with a\n"
                  "character representing the promoted-to piece (n = knight, b = bishop, r = rook, q = queen) (e.g.\n"
                  "e7e8q). To resign, enter \"resign\".")

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

            self.print_board(player, timer, game.history[-1])

            while not game.terminal():
                if game.to_play() == player:
                    start = time.time()
                    while True:
                        action = AlphaZero.lan_to_move_index(input("Move: "))
                        if action in game.legal_actions() or action == MoveIndexing.TOTAL_MOVES or time.time() - start >= timer[player]:
                            break

                    print()

                    timer[player] -= time.time() - start

                    if timer[player] <= 0:
                        if AlphaZero.is_timeout_vs_insufficient_material(game):
                            print("Draw by timeout vs. insufficient material.")
                        else:
                            print("Player lost on time.")
                        break

                    if action == MoveIndexing.TOTAL_MOVES:
                        print("Player lost by resignation.")
                        break

                    timer[player] += increment
                else:
                    start = time.time()
                    root = Node(0)

                    print("Computer is thinking...")
                    print()

                    action = self.run_mcts(game, root, timer[not player] / 20)
                    timer[not player] -= time.time() - start

                    if timer[not player] <= 0:
                        if AlphaZero.is_timeout_vs_insufficient_material(game):
                            print("Draw by timeout vs. insufficient material.")
                        else:
                            print("Computer lost on time.")
                        break

                    timer[not player] += increment

                game.apply(action)

                self.print_board(player, timer, game.history[-1])

            if game.terminal():
                if game.terminal_value(game.to_play()) == 0:
                    print("Draw by " + game.terminal_string + ".")
                elif game.to_play() == player:
                    print("Computer wins by " + game.terminal_string + ".")
                elif game.to_play() == (not player):
                    print("Player wins by " + game.terminal_string + ".")

            print()

            while True:
                again = input("Play again (Y/N)?: ")
                if again.lower() in ['y', 'n']:
                    break
            if again.lower() == 'n':
                break

            print()
            print()

    @staticmethod
    def print_board(player, timer: List[float], state: State):
        if player:
            print("Computer timer: " + str(timer[0]) + "s")
            print("Player timer: " + str(timer[1]) + "s")

            print()

            print("  h g f e d c b a")
            for i in range(7, -1, -1):
                print(8 - i, end=' ')
                for j in range(7, -1, -1):
                    print(state.mailbox[8 * i + j], end=' ')
                print()
            print()
        else:
            print("Player timer: " + str(timer[0]) + "s")
            print("Computer timer: " + str(timer[1]) + "s")

            print()

            print("  a b c d e f g h")
            for i in range(8):
                print(8 - i, end=' ')
                for j in range(8):
                    print(state.mailbox[8 * i + j], end=' ')
                print()
            print()

    def run_mcts(self, game: Game, root: Node, search_time):
        start = time.time()

        if not root.expanded():
            training.evaluate(root, game, self.network)

        training.add_exploration_noise(self.config, root)

        while time.time() - start < search_time:
            node = root
            scratch_game = game.clone()
            search_path = [node]

            while node.expanded():
                action, node = training.select_child(self.config, node)
                scratch_game.apply(action)
                search_path.append(node)

            if scratch_game.terminal():
                value = scratch_game.terminal_value(game.to_play())
            else:
                value = training.evaluate(node, scratch_game, self.network)

            training.backpropagate(search_path, value, scratch_game.to_play())

        return self.select_action(root)

    def select_action(self, root: Node):
        visit_counts = [(child.visit_count, action)
                        for action, child in root.children.items()]
        _, action = max(visit_counts)
        return action
