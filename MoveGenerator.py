from BitboardType import BitboardType
from Color import Color
from State import State
import MoveIndexing

import copy
from enum import IntEnum
import math
import random
import time


class MoveGenerator(object):
    class AttackSetType(IntEnum):
        COLUMN = 0
        DIAGONAL = 1
        WHITE_PAWN = 2
        BLACK_PAWN = 3
        KNIGHT = 4
        KING = 5

    class CaptureUnderpromotionType(IntEnum):
        LEFT = 1
        RIGHT = 2

    class Direction(IntEnum):
        NORTH = 0
        NORTHEAST = 1
        EAST = 2
        SOUTHEAST = 3
        SOUTH = 4
        SOUTHWEST = 5
        WEST = 6
        NORTHWEST = 7

    ATTACK_SETS = [64 * [0] for k in range(len(AttackSetType))]
    RAYS = [64 * [0] for k in range(8)]
    MAGIC_NUMBERS = [64 * [0] for k in range(2)]
    KEY_SIZES = [64 * [0] for k in range(2)]

    BEST_SEED = 4599409325970907659
    IS_INITIALIZED = False

    def __init__(self):
        self.best_time = math.inf
        self.start = 0

    @staticmethod
    def bitstring_to_int(bitstring):
        return int(bitstring[::-1], 2)

    def find_best_seed(self):
        random.seed(MoveGenerator.BEST_SEED)
        self.start = time.time()
        MoveGenerator.initialize_sliding_attack_sets()
        end = time.time()
        self.best_time = end - self.start
        print("Best Seed: " + str(MoveGenerator.BEST_SEED) + " (" + str(self.best_time) + "s)")
        while True:
            seed = random.getrandbits(64)
            self.start = time.time()
            MoveGenerator.initialize_sliding_attack_sets()
            end = time.time()
            if end - self.start < self.best_time:
                MoveGenerator.BEST_SEED = seed
                self.best_time = end - self.start
                print("Best Seed: " + str(MoveGenerator.BEST_SEED) + " (" + str(self.best_time) + "s)")

    @staticmethod
    def generate_castlings(state: State, moves):
        if state.active_color:
            can_kingside_castle = (state.castling_rights & (1 << 2)) > 0
            can_queenside_castle = (state.castling_rights & (1 << 3)) > 0
            kingside_empty_check_bitboard = MoveGenerator.bitstring_to_int("00000110" +
                                                                           "00000000" +
                                                                           "00000000" +
                                                                           "00000000" +
                                                                           "00000000" +
                                                                           "00000000" +
                                                                           "00000000" +
                                                                           "00000000")
            queenside_empty_check_bitboard = MoveGenerator.bitstring_to_int("01110000" +
                                                                            "00000000" +
                                                                            "00000000" +
                                                                            "00000000" +
                                                                            "00000000" +
                                                                            "00000000" +
                                                                            "00000000" +
                                                                            "00000000")
            kingside_attack_check_bitboard = MoveGenerator.bitstring_to_int("00001100" +
                                                                            "00000000" +
                                                                            "00000000" +
                                                                            "00000000" +
                                                                            "00000000" +
                                                                            "00000000" +
                                                                            "00000000" +
                                                                            "00000000")
            queenside_attack_check_bitboard = MoveGenerator.bitstring_to_int("00011000" +
                                                                             "00000000" +
                                                                             "00000000" +
                                                                             "00000000" +
                                                                             "00000000" +
                                                                             "00000000" +
                                                                             "00000000" +
                                                                             "00000000")
            attack_set = MoveGenerator.get_attack_set(state, Color.WHITE)
            i = 0
        else:
            can_kingside_castle = (state.castling_rights & (1 << 0)) > 0
            can_queenside_castle = (state.castling_rights & (1 << 1)) > 0
            kingside_empty_check_bitboard = MoveGenerator.bitstring_to_int("00000000" +
                                                                           "00000000" +
                                                                           "00000000" +
                                                                           "00000000" +
                                                                           "00000000" +
                                                                           "00000000" +
                                                                           "00000000" +
                                                                           "00000110")
            queenside_empty_check_bitboard = MoveGenerator.bitstring_to_int("00000000" +
                                                                            "00000000" +
                                                                            "00000000" +
                                                                            "00000000" +
                                                                            "00000000" +
                                                                            "00000000" +
                                                                            "00000000" +
                                                                            "01110000")
            kingside_attack_check_bitboard = MoveGenerator.bitstring_to_int("00000000" +
                                                                            "00000000" +
                                                                            "00000000" +
                                                                            "00000000" +
                                                                            "00000000" +
                                                                            "00000000" +
                                                                            "00000000" +
                                                                            "00001100")
            queenside_attack_check_bitboard = MoveGenerator.bitstring_to_int("00000000" +
                                                                             "00000000" +
                                                                             "00000000" +
                                                                             "00000000" +
                                                                             "00000000" +
                                                                             "00000000" +
                                                                             "00000000" +
                                                                             "00011000")
            attack_set = MoveGenerator.get_attack_set(state, Color.BLACK)
            i = 7

        if can_kingside_castle and not ~state.bitboards[BitboardType.EMPTY] & kingside_empty_check_bitboard and not kingside_attack_check_bitboard & attack_set:
            moves[MoveIndexing.MOVES_PER_LOCATION * (8 * i + 4) + (
                    MoveIndexing.MOVES_PER_DIRECTION * MoveGenerator.Direction.EAST + 1)] = (i, 4, i, 6)

        if can_queenside_castle and not ~state.bitboards[BitboardType.EMPTY] & queenside_empty_check_bitboard and not queenside_attack_check_bitboard & attack_set:
            moves[MoveIndexing.MOVES_PER_LOCATION * (8 * i + 4) + (
                    MoveIndexing.MOVES_PER_DIRECTION * MoveGenerator.Direction.WEST + 1)] = (i, 4, i, 2)

    @staticmethod
    def generate_column_moves(state: State, moves):
        if state.active_color:
            column_pieces = state.bitboards[BitboardType.BLACK_ROOK] | state.bitboards[BitboardType.BLACK_QUEEN]
            active_bitboard = state.bitboards[BitboardType.BLACK]
        else:
            column_pieces = state.bitboards[BitboardType.WHITE_ROOK] | state.bitboards[BitboardType.WHITE_QUEEN]
            active_bitboard = state.bitboards[BitboardType.WHITE]
        while column_pieces > 0:
            k = int(math.log2(column_pieces & -column_pieces))
            i = k // 8
            j = k % 8
            attack_set = MoveGenerator.get_sliding_attack_set(MoveGenerator.AttackSetType.COLUMN, k,
                                                              state) & ~active_bitboard
            while attack_set > 0:
                n = int(math.log2(attack_set & -attack_set))
                l = n // 8
                m = n % 8
                moves[MoveIndexing.MOVES_PER_LOCATION * k + MoveGenerator.get_sliding_move_index(l - i, m - j)] = (
                    i, j, l, m)
                attack_set -= 1 << n
            column_pieces -= 1 << k

    @staticmethod
    def generate_diagonal_moves(state: State, moves):
        if state.active_color:
            diagonal_pieces = state.bitboards[BitboardType.BLACK_BISHOP] | state.bitboards[BitboardType.BLACK_QUEEN]
            active_bitboard = state.bitboards[BitboardType.BLACK]
        else:
            diagonal_pieces = state.bitboards[BitboardType.WHITE_BISHOP] | state.bitboards[BitboardType.WHITE_QUEEN]
            active_bitboard = state.bitboards[BitboardType.WHITE]
        while diagonal_pieces > 0:
            k = int(math.log2(diagonal_pieces & -diagonal_pieces))
            i = k // 8
            j = k % 8
            attack_set = MoveGenerator.get_sliding_attack_set(MoveGenerator.AttackSetType.DIAGONAL, k,
                                                              state) & ~active_bitboard
            while attack_set > 0:
                n = int(math.log2(attack_set & -attack_set))
                l = n // 8
                m = n % 8
                moves[MoveIndexing.MOVES_PER_LOCATION * k + MoveGenerator.get_sliding_move_index(l - i, m - j)] = (
                    i, j, l, m)
                attack_set -= 1 << n
            diagonal_pieces -= 1 << k

    @staticmethod
    def generate_en_passants(state: State, moves):
        if state.possible_en_passant_target == -1:
            return
        if state.active_color:
            active_pawn_bitboard = state.bitboards[BitboardType.BLACK_PAWN]
            k = 4
            directions = [MoveGenerator.Direction.NORTHWEST, MoveGenerator.Direction.NORTHEAST]
            dk = 1
        else:
            active_pawn_bitboard = state.bitboards[BitboardType.WHITE_PAWN]
            k = 3
            directions = [MoveGenerator.Direction.SOUTHWEST, MoveGenerator.Direction.SOUTHEAST]
            dk = -1
        l = state.possible_en_passant_target
        dl = [-1, 1]
        for m in range(2):
            if l + dl[m] < 0 or l + dl[m] >= 8:
                continue
            if active_pawn_bitboard & 1 << (8 * k + (l + dl[m])):
                moves[MoveIndexing.MOVES_PER_LOCATION * (8 * k + (l + dl[m])) + (
                        MoveIndexing.MOVES_PER_DIRECTION * directions[dl[m] < 0])] = (k, l + dl[m], k + dk, l)

    @staticmethod
    def generate_king_moves(state: State, moves):
        if state.active_color:
            kings = state.bitboards[BitboardType.BLACK_KING]
            active_bitboard = state.bitboards[BitboardType.BLACK]
        else:
            kings = state.bitboards[BitboardType.WHITE_KING]
            active_bitboard = state.bitboards[BitboardType.WHITE]
        dr_to_direction = {
            (-1, 0): MoveGenerator.Direction.NORTH,
            (-1, 1): MoveGenerator.Direction.NORTHEAST,
            (0, 1): MoveGenerator.Direction.EAST,
            (1, 1): MoveGenerator.Direction.SOUTHEAST,
            (1, 0): MoveGenerator.Direction.SOUTH,
            (1, -1): MoveGenerator.Direction.SOUTHWEST,
            (0, -1): MoveGenerator.Direction.WEST,
            (-1, -1): MoveGenerator.Direction.NORTHWEST
        }
        while kings > 0:
            k = int(math.log2(kings & -kings))
            i = k // 8
            j = k % 8
            attack_set = MoveGenerator.ATTACK_SETS[MoveGenerator.AttackSetType.KING][k] & ~active_bitboard
            while attack_set > 0:
                n = int(math.log2(attack_set & -attack_set))
                l = n // 8
                m = n % 8
                moves[MoveIndexing.MOVES_PER_LOCATION * k + (
                        MoveIndexing.MOVES_PER_DIRECTION * dr_to_direction[(l - i, m - j)])] = (i, j, l, m)
                attack_set -= 1 << n
            kings -= 1 << k

    @staticmethod
    def generate_knight_moves(state: State, moves):
        if state.active_color:
            knights = state.bitboards[BitboardType.BLACK_KNIGHT]
            active_bitboard = state.bitboards[BitboardType.BLACK]
        else:
            knights = state.bitboards[BitboardType.WHITE_KNIGHT]
            active_bitboard = state.bitboards[BitboardType.WHITE]
        dr_to_index = {
            (-2, 1): 0,
            (-1, 2): 1,
            (1, 2): 2,
            (2, 1): 3,
            (2, -1): 4,
            (1, -2): 5,
            (-1, -2): 6,
            (-2, -1): 7
        }
        while knights > 0:
            k = int(math.log2(knights & -knights))
            i = k // 8
            j = k % 8
            attack_set = MoveGenerator.ATTACK_SETS[MoveGenerator.AttackSetType.KNIGHT][k] & ~active_bitboard
            while attack_set > 0:
                n = int(math.log2(attack_set & -attack_set))
                l = n // 8
                m = n % 8
                if (l - i, m - j) == (-2, 0):
                    print(k)
                    print(n)
                    MoveGenerator.print_bitboard(attack_set)
                moves[MoveIndexing.MOVES_PER_LOCATION * k + (
                        MoveIndexing.KNIGHT_MOVE_INDEX + dr_to_index[(l - i, m - j)])] = (
                    i, j, l, m)
                attack_set -= 1 << n
            knights -= 1 << k

    @staticmethod
    def generate_moves(state: State):
        if not MoveGenerator.IS_INITIALIZED:
            MoveGenerator.initialize()

        moves = {}
        MoveGenerator.generate_pawn_one_forward_moves(state, moves)
        MoveGenerator.generate_pawn_two_forward_moves(state, moves)
        MoveGenerator.generate_pawn_captures(state, moves)
        MoveGenerator.generate_en_passants(state, moves)
        MoveGenerator.generate_knight_moves(state, moves)
        MoveGenerator.generate_king_moves(state, moves)
        MoveGenerator.generate_castlings(state, moves)
        MoveGenerator.generate_column_moves(state, moves)
        MoveGenerator.generate_diagonal_moves(state, moves)

        to_remove = []
        hash_code = copy.deepcopy(state.hash_code)
        for action, move_info in moves.items():
            state.make_move(action, move_info[0], move_info[1], move_info[2], move_info[3])
            if state.active_color:
                color = Color.BLACK
                king_bitboard = state.bitboards[BitboardType.WHITE_KING]
            else:
                color = Color.WHITE
                king_bitboard = state.bitboards[BitboardType.BLACK_KING]
            if king_bitboard & MoveGenerator.get_attack_set(state, color):
                to_remove.append(action)
            state.set_state(hash_code)

        for action in to_remove:
            moves.pop(action)

        return moves

    @staticmethod
    def generate_pawn_captures(state: State, moves):
        if state.active_color:
            pawns = state.bitboards[BitboardType.BLACK_PAWN]
            attack_sets = MoveGenerator.ATTACK_SETS[MoveGenerator.AttackSetType.BLACK_PAWN]
            inactive_bitboard = state.bitboards[BitboardType.WHITE]
            directions = [MoveGenerator.Direction.NORTHWEST, MoveGenerator.Direction.NORTHEAST]
            promotion_l = 7
        else:
            pawns = state.bitboards[BitboardType.WHITE_PAWN]
            attack_sets = MoveGenerator.ATTACK_SETS[MoveGenerator.AttackSetType.WHITE_PAWN]
            inactive_bitboard = state.bitboards[BitboardType.BLACK]
            directions = [MoveGenerator.Direction.SOUTHWEST, MoveGenerator.Direction.SOUTHEAST]
            promotion_l = 0
        while pawns > 0:
            k = int(math.log2(pawns & -pawns))
            i = k // 8
            j = k % 8
            attack_set = attack_sets[k] & inactive_bitboard
            dj_to_capture_underpromotion_type = {
                -1: MoveGenerator.CaptureUnderpromotionType.LEFT,
                1: MoveGenerator.CaptureUnderpromotionType.RIGHT
            }
            while attack_set > 0:
                n = int(math.log2(attack_set & -attack_set))
                l = n // 8
                m = n % 8
                moves[MoveIndexing.MOVES_PER_LOCATION * k + (MoveIndexing.MOVES_PER_DIRECTION * directions[m > j])] = (
                    i, j, l, m)
                if l == promotion_l:
                    for o in range(3):
                        moves[MoveIndexing.MOVES_PER_LOCATION * k + (
                                MoveIndexing.UNDERPROMOTION_INDEX + 3 * dj_to_capture_underpromotion_type[
                            (m - j)] + o)] = (
                            i, j, l, m)
                attack_set -= 1 << n
            pawns -= 1 << k

    @staticmethod
    def generate_pawn_one_forward_moves(state: State, moves):
        if state.active_color:
            pawns = state.bitboards[BitboardType.EMPTY] >> 8 & state.bitboards[BitboardType.BLACK_PAWN]
            direction = MoveGenerator.Direction.SOUTH
            di = 1
            promotion_i = 6
        else:
            pawns = state.bitboards[BitboardType.EMPTY] << 8 & state.bitboards[BitboardType.WHITE_PAWN]
            direction = MoveGenerator.Direction.NORTH
            di = -1
            promotion_i = 1
        while pawns > 0:
            k = int(math.log2(pawns & -pawns))
            i = k // 8
            j = k % 8
            moves[MoveIndexing.MOVES_PER_LOCATION * k + (MoveIndexing.MOVES_PER_DIRECTION * direction)] = (
                i, j, i + di, j)
            if i == promotion_i:
                for l in range(3):
                    moves[MoveIndexing.MOVES_PER_LOCATION * k + (MoveIndexing.UNDERPROMOTION_INDEX + l)] = (
                        i, j, i + di, j)
            pawns -= 1 << k

    @staticmethod
    def generate_pawn_two_forward_moves(state: State, moves):
        if state.active_color:
            pawns = (state.bitboards[BitboardType.EMPTY] >> 8) \
                    & (state.bitboards[BitboardType.EMPTY] >> 16) \
                    & state.bitboards[BitboardType.BLACK_PAWN] \
                    & MoveGenerator.bitstring_to_int("00000000" +
                                                     "11111111" +
                                                     "00000000" +
                                                     "00000000" +
                                                     "00000000" +
                                                     "00000000" +
                                                     "00000000" +
                                                     "00000000")
            direction = MoveGenerator.Direction.SOUTH
            di = 1
        else:
            pawns = (state.bitboards[BitboardType.EMPTY] << 8) \
                    & (state.bitboards[BitboardType.EMPTY] << 16) \
                    & state.bitboards[BitboardType.WHITE_PAWN] \
                    & MoveGenerator.bitstring_to_int("00000000" +
                                                     "00000000" +
                                                     "00000000" +
                                                     "00000000" +
                                                     "00000000" +
                                                     "00000000" +
                                                     "11111111" +
                                                     "00000000")
            direction = MoveGenerator.Direction.NORTH
            di = -1
        while pawns > 0:
            k = int(math.log2(pawns & -pawns))
            i = k // 8
            j = k % 8
            moves[MoveIndexing.MOVES_PER_LOCATION * k + (MoveIndexing.MOVES_PER_DIRECTION * direction + 1)] = (
                i, j, i + 2 * di, j)
            pawns -= 1 << k

    @staticmethod
    def generate_sliding_attack_set(masked_blockers, attack_set_type: AttackSetType, k):
        if attack_set_type == MoveGenerator.AttackSetType.COLUMN:
            rays = [MoveGenerator.RAYS[MoveGenerator.Direction.NORTH],
                    MoveGenerator.RAYS[MoveGenerator.Direction.EAST],
                    MoveGenerator.RAYS[MoveGenerator.Direction.SOUTH],
                    MoveGenerator.RAYS[MoveGenerator.Direction.WEST]]
        else:
            rays = [MoveGenerator.RAYS[MoveGenerator.Direction.NORTHEAST],
                    MoveGenerator.RAYS[MoveGenerator.Direction.SOUTHEAST],
                    MoveGenerator.RAYS[MoveGenerator.Direction.SOUTHWEST],
                    MoveGenerator.RAYS[MoveGenerator.Direction.NORTHWEST]]
        masked_blockers = [masked_blockers & rays[0][k],
                           masked_blockers & rays[1][k],
                           masked_blockers & rays[2][k],
                           masked_blockers & rays[3][k]]
        blocked_rays = 4 * [0]
        if masked_blockers[0] == 0:
            blocked_rays[0] = 0
        else:
            blocked_rays[0] = rays[0][int(math.log2(masked_blockers[0]))]
        if masked_blockers[1] == 0:
            blocked_rays[1] = 0
        else:
            blocked_rays[1] = rays[1][int(math.log2(masked_blockers[1] & -masked_blockers[1]))]
        if masked_blockers[2] == 0:
            blocked_rays[2] = 0
        else:
            blocked_rays[2] = rays[2][int(math.log2(masked_blockers[2] & -masked_blockers[2]))]
        if masked_blockers[3] == 0:
            blocked_rays[3] = 0
        else:
            blocked_rays[3] = rays[3][int(math.log2(masked_blockers[3]))]
        attack_set = 0
        for l in range(4):
            attack_set |= rays[l][k] & ~blocked_rays[l]
        return attack_set

    @staticmethod
    def generate_sliding_attack_sets(masked_blockers, k, attack_sets,
                                     attack_set_type: AttackSetType, l):
        if k == 64:
            attack_sets[masked_blockers] = MoveGenerator.generate_sliding_attack_set(masked_blockers, attack_set_type,
                                                                                     l)
            return
        if not masked_blockers & (1 << k):
            MoveGenerator.generate_sliding_attack_sets(masked_blockers, k + 1, attack_sets, attack_set_type, l)
            return
        masked_blockers -= 1 << k
        MoveGenerator.generate_sliding_attack_sets(masked_blockers, k + 1, attack_sets, attack_set_type, l)
        masked_blockers += 1 << k
        MoveGenerator.generate_sliding_attack_sets(masked_blockers, k + 1, attack_sets, attack_set_type, l)

    @staticmethod
    def get_attack_set(state: State, color: Color):
        if color == Color.BLACK:
            pawn_character = 'p'
            knight_character = 'n'
            bishop_character = 'b'
            rook_character = 'r'
            queen_character = 'q'
            king_character = 'k'
            pawn_attack_set = MoveGenerator.ATTACK_SETS[MoveGenerator.AttackSetType.BLACK_PAWN]
        else:
            pawn_character = 'P'
            knight_character = 'N'
            bishop_character = 'B'
            rook_character = 'R'
            queen_character = 'Q'
            king_character = 'K'
            pawn_attack_set = MoveGenerator.ATTACK_SETS[MoveGenerator.AttackSetType.WHITE_PAWN]
        attack_set = 0
        for k in range(64):
            if state.mailbox[k] == pawn_character:
                attack_set |= pawn_attack_set[k]
            if state.mailbox[k] == knight_character:
                attack_set |= MoveGenerator.ATTACK_SETS[MoveGenerator.AttackSetType.KNIGHT][k]
            if state.mailbox[k] == bishop_character or state.mailbox[k] == queen_character:
                attack_set |= MoveGenerator.get_sliding_attack_set(MoveGenerator.AttackSetType.DIAGONAL, k, state)
            if state.mailbox[k] == rook_character or state.mailbox[k] == queen_character:
                attack_set |= MoveGenerator.get_sliding_attack_set(MoveGenerator.AttackSetType.COLUMN, k, state)
            if state.mailbox[k] == king_character:
                attack_set |= MoveGenerator.ATTACK_SETS[MoveGenerator.AttackSetType.KING][k]
        return attack_set

    @staticmethod
    def get_sliding_attack_set(attack_set_type: AttackSetType, k, state: State):
        if attack_set_type == MoveGenerator.AttackSetType.COLUMN:
            masked_blockers = ~state.bitboards[BitboardType.EMPTY] \
                              & (MoveGenerator.RAYS[MoveGenerator.Direction.NORTH][k]
                                 & ~MoveGenerator.bitstring_to_int("11111111" +
                                                                   "00000000" +
                                                                   "00000000" +
                                                                   "00000000" +
                                                                   "00000000" +
                                                                   "00000000" +
                                                                   "00000000" +
                                                                   "00000000")
                                 | MoveGenerator.RAYS[MoveGenerator.Direction.EAST][k]
                                 & ~MoveGenerator.bitstring_to_int("00000001" +
                                                                   "00000001" +
                                                                   "00000001" +
                                                                   "00000001" +
                                                                   "00000001" +
                                                                   "00000001" +
                                                                   "00000001" +
                                                                   "00000001")
                                 | MoveGenerator.RAYS[MoveGenerator.Direction.SOUTH][k]
                                 & ~MoveGenerator.bitstring_to_int("00000000" +
                                                                   "00000000" +
                                                                   "00000000" +
                                                                   "00000000" +
                                                                   "00000000" +
                                                                   "00000000" +
                                                                   "00000000" +
                                                                   "11111111")
                                 | MoveGenerator.RAYS[MoveGenerator.Direction.WEST][k]
                                 & ~MoveGenerator.bitstring_to_int("10000000" +
                                                                   "10000000" +
                                                                   "10000000" +
                                                                   "10000000" +
                                                                   "10000000" +
                                                                   "10000000" +
                                                                   "10000000" +
                                                                   "10000000"))
        else:
            masked_blockers = ~state.bitboards[BitboardType.EMPTY] \
                              & (MoveGenerator.RAYS[MoveGenerator.Direction.NORTHEAST][k]
                                 & ~MoveGenerator.bitstring_to_int("11111111" +
                                                                   "00000001" +
                                                                   "00000001" +
                                                                   "00000001" +
                                                                   "00000001" +
                                                                   "00000001" +
                                                                   "00000001" +
                                                                   "00000001")
                                 | MoveGenerator.RAYS[MoveGenerator.Direction.SOUTHEAST][k]
                                 & ~MoveGenerator.bitstring_to_int("00000001" +
                                                                   "00000001" +
                                                                   "00000001" +
                                                                   "00000001" +
                                                                   "00000001" +
                                                                   "00000001" +
                                                                   "00000001" +
                                                                   "11111111")
                                 | MoveGenerator.RAYS[MoveGenerator.Direction.SOUTHWEST][k]
                                 & ~MoveGenerator.bitstring_to_int("10000000" +
                                                                   "10000000" +
                                                                   "10000000" +
                                                                   "10000000" +
                                                                   "10000000" +
                                                                   "10000000" +
                                                                   "10000000" +
                                                                   "11111111")
                                 | MoveGenerator.RAYS[MoveGenerator.Direction.NORTHWEST][k]
                                 & ~MoveGenerator.bitstring_to_int("11111111" +
                                                                   "10000000" +
                                                                   "10000000" +
                                                                   "10000000" +
                                                                   "10000000" +
                                                                   "10000000" +
                                                                   "10000000" +
                                                                   "10000000"))
        key = ((masked_blockers * MoveGenerator.MAGIC_NUMBERS[attack_set_type][k]) % (1 << 64)) >> (
                64 - MoveGenerator.KEY_SIZES[attack_set_type][k])
        return MoveGenerator.ATTACK_SETS[attack_set_type][k][key]

    @staticmethod
    def get_sliding_move_index(di, dj):
        if di < 0 and dj == 0:
            direction = MoveGenerator.Direction.NORTH
        elif di < 0 and dj > 0:
            direction = MoveGenerator.Direction.NORTHEAST
        elif di == 0 and dj > 0:
            direction = MoveGenerator.Direction.EAST
        elif di > 0 and dj > 0:
            direction = MoveGenerator.Direction.SOUTHEAST
        elif di > 0 and dj == 0:
            direction = MoveGenerator.Direction.SOUTH
        elif di > 0 and dj < 0:
            direction = MoveGenerator.Direction.SOUTHWEST
        elif di == 0 and dj < 0:
            direction = MoveGenerator.Direction.WEST
        else:
            direction = MoveGenerator.Direction.NORTHWEST
        return MoveIndexing.MOVES_PER_DIRECTION * direction + max(abs(di), abs(dj)) - 1

    @staticmethod
    def initialize():
        if MoveGenerator.IS_INITIALIZED:
            return
        random.seed(MoveGenerator.BEST_SEED)
        MoveGenerator.initialize_pawn_attack_sets()
        MoveGenerator.initialize_knight_attack_sets()
        MoveGenerator.initialize_king_attack_sets()
        MoveGenerator.initialize_rays()
        MoveGenerator.initialize_sliding_attack_sets()
        MoveGenerator.IS_INITIALIZED = True

    @staticmethod
    def initialize_east_rays():
        for i in range(8):
            ray = 0
            for j in range(7, -1, -1):
                MoveGenerator.RAYS[MoveGenerator.Direction.EAST][8 * i + j] = ray
                ray += 1 << (8 * i + j)

    @staticmethod
    def initialize_king_attack_sets():
        di = [-1, -1, 0, 1, 1, 1, 0, -1]
        dj = [0, 1, 1, 1, 0, -1, -1, -1]
        for i in range(8):
            for j in range(8):
                for k in range(8):
                    if i + di[k] < 0 or i + di[k] >= 8 or j + dj[k] < 0 or j + dj[k] >= 8:
                        continue
                    MoveGenerator.ATTACK_SETS[MoveGenerator.AttackSetType.KING][8 * i + j] += 1 << (
                            8 * (i + di[k]) + (j + dj[k]))

    @staticmethod
    def initialize_knight_attack_sets():
        di = [-2, -1, 1, 2, 2, 1, -1, -2]
        dj = [1, 2, 2, 1, -1, -2, -2, -1]
        for i in range(8):
            for j in range(8):
                for k in range(8):
                    if i + di[k] < 0 or i + di[k] >= 8 or j + dj[k] < 0 or j + dj[k] >= 8:
                        continue
                    MoveGenerator.ATTACK_SETS[MoveGenerator.AttackSetType.KNIGHT][8 * i + j] += 1 << (
                            8 * (i + di[k]) + (j + dj[k]))

    @staticmethod
    def initialize_north_rays():
        for j in range(8):
            ray = 0
            for i in range(8):
                MoveGenerator.RAYS[MoveGenerator.Direction.NORTH][8 * i + j] = ray
                ray += 1 << (8 * i + j)

    @staticmethod
    def initialize_northeast_rays():
        for j in range(8):
            ray = 0
            i = 0
            k = 0
            while i + k < 8 and j - k >= 0:
                MoveGenerator.RAYS[MoveGenerator.Direction.NORTHEAST][8 * (i + k) + (j - k)] = ray
                ray += 1 << (8 * (i + k) + (j - k))
                k += 1
        for i in range(1, 8):
            ray = 0
            j = 7
            k = 0
            while i + k < 8 and j - k >= 0:
                MoveGenerator.RAYS[MoveGenerator.Direction.NORTHEAST][8 * (i + k) + (j - k)] = ray
                ray += 1 << (8 * (i + k) + (j - k))
                k += 1

    @staticmethod
    def initialize_northwest_rays():
        for j in range(8):
            rays = 0
            i = 0
            k = 0
            while i + k < 8 and j + k < 8:
                MoveGenerator.RAYS[MoveGenerator.Direction.NORTHWEST][8 * (i + k) + (j + k)] = rays
                rays += 1 << (8 * (i + k) + (j + k))
                k += 1
        for i in range(1, 8):
            rays = 0
            j = 0
            k = 0
            while i + k < 8 and j + k < 8:
                MoveGenerator.RAYS[MoveGenerator.Direction.NORTHWEST][8 * (i + k) + (j + k)] = rays
                rays += 1 << (8 * (i + k) + (j + k))
                k += 1

    @staticmethod
    def initialize_pawn_attack_sets():
        dj = [-1, 1]
        for i in range(1, 8):
            for j in range(8):
                for k in range(2):
                    if j + dj[k] < 0 or j + dj[k] >= 8:
                        continue
                    MoveGenerator.ATTACK_SETS[MoveGenerator.AttackSetType.WHITE_PAWN][8 * i + j] += 1 << (
                            8 * (i - 1) + (j + dj[k]))
        for i in range(7):
            for j in range(8):
                for k in range(2):
                    if j + dj[k] < 0 or j + dj[k] >= 8:
                        continue
                    MoveGenerator.ATTACK_SETS[MoveGenerator.AttackSetType.BLACK_PAWN][8 * i + j] += 1 << (
                            8 * (i + 1) + (j + dj[k]))

    @staticmethod
    def initialize_rays():
        MoveGenerator.initialize_north_rays()
        MoveGenerator.initialize_northeast_rays()
        MoveGenerator.initialize_east_rays()
        MoveGenerator.initialize_southeast_rays()
        MoveGenerator.initialize_south_rays()
        MoveGenerator.initialize_southwest_rays()
        MoveGenerator.initialize_west_rays()
        MoveGenerator.initialize_northwest_rays()

    @staticmethod
    def initialize_sliding_attack_sets():
        for k in range(64):
            masked_blockers = MoveGenerator.RAYS[MoveGenerator.Direction.NORTH][k] \
                              & ~MoveGenerator.bitstring_to_int("11111111" +
                                                                "00000000" +
                                                                "00000000" +
                                                                "00000000" +
                                                                "00000000" +
                                                                "00000000" +
                                                                "00000000" +
                                                                "00000000") \
                              | MoveGenerator.RAYS[MoveGenerator.Direction.EAST][k] \
                              & ~MoveGenerator.bitstring_to_int("00000001" +
                                                                "00000001" +
                                                                "00000001" +
                                                                "00000001" +
                                                                "00000001" +
                                                                "00000001" +
                                                                "00000001" +
                                                                "00000001") \
                              | MoveGenerator.RAYS[MoveGenerator.Direction.SOUTH][k] \
                              & ~MoveGenerator.bitstring_to_int("00000000" +
                                                                "00000000" +
                                                                "00000000" +
                                                                "00000000" +
                                                                "00000000" +
                                                                "00000000" +
                                                                "00000000" +
                                                                "11111111") \
                              | MoveGenerator.RAYS[MoveGenerator.Direction.WEST][k] \
                              & ~MoveGenerator.bitstring_to_int("10000000" +
                                                                "10000000" +
                                                                "10000000" +
                                                                "10000000" +
                                                                "10000000" +
                                                                "10000000" +
                                                                "10000000" +
                                                                "10000000")
            attack_sets = {}
            MoveGenerator.generate_sliding_attack_sets(masked_blockers, 0, attack_sets,
                                                       MoveGenerator.AttackSetType.COLUMN,
                                                       k)
            key_size = 3
            while True:
                magic_number = random.getrandbits(64) & random.getrandbits(64) & random.getrandbits(64)
                if MoveGenerator.is_there_collision(MoveGenerator.AttackSetType.COLUMN, k, magic_number, key_size,
                                                    attack_sets):
                    break
                key_size = min(key_size + 1, 12)
            masked_blockers = MoveGenerator.RAYS[MoveGenerator.Direction.NORTHEAST][k] \
                              & ~MoveGenerator.bitstring_to_int("11111111" +
                                                                "00000001" +
                                                                "00000001" +
                                                                "00000001" +
                                                                "00000001" +
                                                                "00000001" +
                                                                "00000001" +
                                                                "00000001") \
                              | MoveGenerator.RAYS[MoveGenerator.Direction.SOUTHEAST][k] \
                              & ~MoveGenerator.bitstring_to_int("00000001" +
                                                                "00000001" +
                                                                "00000001" +
                                                                "00000001" +
                                                                "00000001" +
                                                                "00000001" +
                                                                "00000001" +
                                                                "11111111") \
                              | MoveGenerator.RAYS[MoveGenerator.Direction.SOUTHWEST][k] \
                              & ~MoveGenerator.bitstring_to_int("10000000" +
                                                                "10000000" +
                                                                "10000000" +
                                                                "10000000" +
                                                                "10000000" +
                                                                "10000000" +
                                                                "10000000" +
                                                                "11111111") \
                              | MoveGenerator.RAYS[MoveGenerator.Direction.NORTHWEST][k] \
                              & ~MoveGenerator.bitstring_to_int("11111111" +
                                                                "10000000" +
                                                                "10000000" +
                                                                "10000000" +
                                                                "10000000" +
                                                                "10000000" +
                                                                "10000000" +
                                                                "10000000")
            attack_sets = {}
            MoveGenerator.generate_sliding_attack_sets(masked_blockers, 0, attack_sets,
                                                       MoveGenerator.AttackSetType.DIAGONAL, k)
            key_size = 3
            while True:
                magic_number = random.getrandbits(64) & random.getrandbits(64) & random.getrandbits(64)
                if MoveGenerator.is_there_collision(MoveGenerator.AttackSetType.DIAGONAL, k, magic_number, key_size,
                                                    attack_sets):
                    break
                key_size = min(key_size + 1, 12)

    @staticmethod
    def initialize_south_rays():
        for j in range(8):
            rays = 0
            for i in range(7, -1, -1):
                MoveGenerator.RAYS[MoveGenerator.Direction.SOUTH][8 * i + j] = rays
                rays += 1 << (8 * i + j)

    @staticmethod
    def initialize_southeast_rays():
        for j in range(8):
            ray = 0
            i = 7
            k = 0
            while i - k >= 0 and j - k >= 0:
                MoveGenerator.RAYS[MoveGenerator.Direction.SOUTHEAST][8 * (i - k) + (j - k)] = ray
                ray += 1 << (8 * (i - k) + (j - k))
                k += 1
        for i in range(7):
            ray = 0
            j = 7
            k = 0
            while i - k >= 0 and j - k >= 0:
                MoveGenerator.RAYS[MoveGenerator.Direction.SOUTHEAST][8 * (i - k) + (j - k)] = ray
                ray += 1 << (8 * (i - k) + (j - k))
                k += 1

    @staticmethod
    def initialize_southwest_rays():
        for j in range(8):
            ray = 0
            i = 7
            k = 0
            while i - k >= 0 and j + k < 8:
                MoveGenerator.RAYS[MoveGenerator.Direction.SOUTHWEST][8 * (i - k) + (j + k)] = ray
                ray += 1 << (8 * (i - k) + (j + k))
                k += 1
        for i in range(7):
            ray = 0
            j = 0
            k = 0
            while i - k >= 0 and j + k < 8:
                MoveGenerator.RAYS[MoveGenerator.Direction.SOUTHWEST][8 * (i - k) + (j + k)] = ray
                ray += 1 << (8 * (i - k) + (j + k))
                k += 1

    @staticmethod
    def initialize_west_rays():
        for i in range(8):
            ray = 0
            for j in range(8):
                MoveGenerator.RAYS[MoveGenerator.Direction.WEST][8 * i + j] = ray
                ray += 1 << (8 * i + j)

    @staticmethod
    def is_there_collision(attack_set_type: AttackSetType, k, magic_number,
                           key_size, attack_sets):
        MoveGenerator.ATTACK_SETS[attack_set_type][k] = (1 << key_size) * [0]
        for masked_blockers, attack_set in attack_sets.items():
            key = ((masked_blockers * magic_number) % (1 << 64)) >> (64 - key_size)
            stored_attack_set = MoveGenerator.ATTACK_SETS[attack_set_type][k][key]
            if stored_attack_set > 0 and stored_attack_set != attack_set:
                return False
            MoveGenerator.ATTACK_SETS[attack_set_type][k][key] = attack_set
        MoveGenerator.MAGIC_NUMBERS[attack_set_type][k] = magic_number
        MoveGenerator.KEY_SIZES[attack_set_type][k] = key_size
        return True

    @staticmethod
    def move_to_lan(state: State, move_index, i, j, k, l):
        is_promotion = state.mailbox[8 * i + j].lower() == 'p' and (k == 0 or k == 7)
        if not is_promotion:
            return chr(j + ord('a')) + str(8 - i) + chr(l + ord('a')) + str(8 - k)
        elif move_index < MoveIndexing.UNDERPROMOTION_INDEX:
            return chr(j + ord('a')) + str(8 - i) + chr(l + ord('a')) + str(8 - k) + 'q'
        elif (move_index - MoveIndexing.UNDERPROMOTION_INDEX) % 3 == 0:
            return chr(j + ord('a')) + str(8 - i) + chr(l + ord('a')) + str(8 - k) + 'r'
        elif (move_index - MoveIndexing.UNDERPROMOTION_INDEX) % 3 == 1:
            return chr(j + ord('a')) + str(8 - i) + chr(l + ord('a')) + str(8 - k) + 'b'
        else:
            return chr(j + ord('a')) + str(8 - i) + chr(l + ord('a')) + str(8 - k) + 'n'

    @staticmethod
    def perft(state: State, depth):
        if depth == 0:
            return 1
        moves = MoveGenerator.generate_moves(state)
        if depth == 1:
            return len(moves)
        total_nodes = 0
        for action, move_info in moves.items():
            hash_code = copy.deepcopy(state.hash_code)
            state.make_move(action % MoveIndexing.MOVES_PER_LOCATION, move_info[0], move_info[1], move_info[2], move_info[3])
            nodes = MoveGenerator.perft(state, depth - 1)
            state.set_state(hash_code)
            total_nodes += nodes
        return total_nodes

    @staticmethod
    def perft_root(state: State, depth):
        moves = MoveGenerator.generate_moves(state)
        total_nodes = 0
        for action, move_info in moves.items():
            hash_code = copy.deepcopy(state.hash_code)
            state.make_move(action % MoveIndexing.MOVES_PER_LOCATION, move_info[0], move_info[1], move_info[2], move_info[3])
            nodes = MoveGenerator.perft(state, depth - 1)
            state.set_state(hash_code)
            total_nodes += nodes
            print(MoveGenerator.move_to_lan(state, action % MoveIndexing.MOVES_PER_LOCATION, move_info[0], move_info[1], move_info[2], move_info[3]) + ": " + str(nodes))
        print()
        print("Nodes searched: " + str(total_nodes))
        print()

    @staticmethod
    def print_bitboard(bitboard):
        for i in range(8):
            for j in range(8):
                if bitboard & 1 << (8 * i + j):
                    print('1', end='')
                else:
                    print('0', end='')
            print()
        print()
