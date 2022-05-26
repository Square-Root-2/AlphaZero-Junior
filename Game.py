from BitboardType import BitboardType
from Color import Color
from MoveGenerator import MoveGenerator
from State import State

from enum import IntEnum
import tensorflow as tf

class Game(object):

    class FeaturePlaneType(IntEnum):
        WHITE_PAWN = 0
        WHITE_KNIGHT = 1
        WHITE_BISHOP = 2
        WHITE_ROOK = 3
        WHITE_QUEEN = 4
        WHITE_KING = 5
        BLACK_PAWN = 6
        BLACK_KNIGHT = 7
        BLACK_BISHOP = 8
        BLACK_ROOK = 9
        BLACK_QUEEN = 10
        BLACK_KING = 11
        ACTIVE_COLOR = 12
        CASTLING_RIGHTS = 13
        POSSIBLE_EN_PASSANT_TARGET = 14
        HALFMOVE_CLOCK = 15

    def __init__(self, history=None):
        self.history = history or [State()]
        self.child_visits = []
        self.num_actions = 4672

        self.actions = None

        self.counts = {}
        for state in self.history:
            if state.get_key() not in self.counts.keys():
                self.counts[state.get_key()] = 0
            self.counts[state.get_key()] += 1

        self.is_terminal = None
        self.z = None

    def terminal(self):
        if self.is_terminal is not None:
            return self.is_terminal

        # checkmate or stalemate
        self.legal_actions()

        if len(self.actions) == 0:
            if self.history[-1].active_color:
                king_bitboard = self.history[-1].bitboards[BitboardType.BLACK]
                attack_set = MoveGenerator.get_attack_set(self.history[-1], Color.WHITE)
            else:
                king_bitboard = self.history[-1].bitboards[BitboardType.WHITE]
                attack_set = MoveGenerator.get_attack_set(self.history[-1], Color.BLACK)

            self.is_terminal = True

            if king_bitboard & attack_set:  # checkmate
                self.z = -1
            else:  # stalemate
                self.z = 0

            return True

        # insufficient material
        if not self.history[-1].bitboards[BitboardType.WHITE_PAWN] | self.history[-1].bitboards[BitboardType.BLACK_PAWN] \
               | self.history[-1].bitboards[BitboardType.WHITE_ROOK] | self.history[-1].bitboards[
                   BitboardType.BLACK_ROOK] \
               | self.history[-1].bitboards[BitboardType.WHITE_QUEEN] | self.history[-1].bitboards[
                   BitboardType.BLACK_QUEEN]:  # no pawns, rooks, or queens
            white_knight_count = self.history[-1].bitboards[BitboardType.WHITE_KNIGHT].bit_count()
            white_bishop_count = self.history[-1].bitboards[BitboardType.WHITE_BISHOP].bit_count()
            black_knight_count = self.history[-1].bitboards[BitboardType.BLACK_KNIGHT].bit_count()
            black_bishop_count = self.history[-1].bitboards[BitboardType.BLACK_BISHOP].bit_count()

            # king (+ minor piece) vs king (+ minor piece)
            if white_knight_count + white_bishop_count <= 1 and black_knight_count + black_bishop_count <= 1:
                self.is_terminal = True
                self.z = 0
                return True

            # king + two knights vs king
            if white_knight_count + white_bishop_count + black_knight_count + black_bishop_count == 2 and (
                    white_knight_count == 2 or black_knight_count == 2):
                self.is_terminal = True
                self.z = 0
                return True

        # 50-move rule
        if self.history[-1].halfmove_clock >= 50:
            self.is_terminal = True
            self.z = 0
            return True

        # repetition
        if self.counts.get(self.history[-1].get_key(), 0) >= 3:
            self.is_terminal = True
            self.z = 0
            return True

        self.is_terminal = False
        return False

    def terminal_value(self, to_play):
        assert self.terminal()
        if to_play == self.history[-1].active_color:
            return self.z
        return -self.z

    def legal_actions(self):
        if self.actions is None:
            self.actions = MoveGenerator.generate_moves(self.history[-1])
        return list(self.actions.keys())

    def clone(self):
        return Game(list(self.history))

    def apply(self, action):
        self.legal_actions()
        state = self.history[-1].clone()
        move_info = self.actions[action]
        state.make_move(action, move_info[0], move_info[1], move_info[2], move_info[3])
        self.history.append(state)

        self.actions = None

        if self.history[-1].get_key() not in self.counts.keys():
            self.counts[self.history[-1].get_key()] = 0

        self.counts[self.history[-1].get_key()] += 1

        self.is_terminal = None
        self.z = None

    def store_search_statistics(self, root):
        sum_visits = sum(child.visit_count for child in root.children.values())
        self.child_visits.append([
            root.children[a].visit_count / sum_visits if a in root.children else 0
            for a in range(self.num_actions)
        ])

    def make_image(self, state_index: int):
        image = [[16 * [0] for l in range(8)] for k in range(8)]

        for i in range(8):
            for j in range(8):
                # feature planes for piece types
                for k in range(Game.FeaturePlaneType.ACTIVE_COLOR):
                    image[i][j][k] = max(self.history[state_index].bitboards[k] & (1 << (8 * i + j)), 1)

                # constant feature plane for active color
                if self.history[state_index].active_color:
                    image[i][j][Game.FeaturePlaneType.ACTIVE_COLOR] = 1

                # feature plane for possible en passant target
                if j == self.history[state_index].possible_en_passant_target:
                    image[i][j][Game.FeaturePlaneType.POSSIBLE_EN_PASSANT_TARGET] = 1

                # (constant) feature plane for halfmove clock
                image[i][j][Game.FeaturePlaneType.HALFMOVE_CLOCK] = min(self.history[state_index].halfmove_clock / 50, 1)

        # feature plane for castling rights
        i = [7, 7, 0, 0]
        j = [7, 0, 7, 0]
        for k in range(4):
            if self.history[state_index].castling_rights & (1 << k):
                image[i[k]][j[k]][Game.FeaturePlaneType.CASTLING_RIGHTS] = 1

        return tf.expand_dims(tf.constant(image), axis=0)

    def make_target(self, state_index: int):
        return (self.terminal_value(state_index % 2),
                self.child_visits[state_index])

    def to_play(self):
        return (len(self.history) + 1) % 2
