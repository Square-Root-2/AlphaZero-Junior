import random

from BitboardType import BitboardType
from State import State

from enum import IntEnum
import math


class MoveGenerator(object):
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

    class MagicBitboardType(IntEnum):
        COLUMN = 0
        DIAGONAL = 1

    class PawnAttackSetType(IntEnum):
        WHITE = 0
        BLACK = 1

    def __init__(self):
        self.MOVES_PER_LOCATION = 73
        self.KNIGHT_MOVE_INDEX = 56
        self.UNDERPROMOTION_INDEX = 64

        self.pawn_attack_sets = [64 * [0] for k in range(2)]
        self.knight_attack_sets = 64 * [0]
        self.king_attack_sets = 64 * [0]
        self.rays = [64 * [0] for k in range(8)]
        self.magic_bitboards = [64 * [[]] for k in range(2)]

        self.initialize_pawn_attack_sets()
        self.initialize_knight_attack_sets()
        self.initialize_king_attack_sets()
        self.initialize_sliding_piece_attack_sets()
        self.initialize_magic_bitboards()

    def generate_castlings(self, state: State) -> dict:
        moves = {}
        if state.active_color and state.black_kingside:
            moves[self.MOVES_PER_LOCATION * 4 + (7 * self.Direction.EAST + 1)] = (0, 4, 0, 6)
        elif not state.active_color and state.white_kingside:
            moves[self.MOVES_PER_LOCATION * 60 + (7 * self.Direction.EAST + 1)] = (7, 4, 7, 6)
        if state.active_color and state.black_queenside:
            moves[self.MOVES_PER_LOCATION * 4 + (7 * self.Direction.WEST + 1)] = (0, 4, 0, 2)
        elif not state.active_color and state.white_queenside:
            moves[self.MOVES_PER_LOCATION * 60 + (7 * self.Direction.WEST + 1)] = (7, 4, 7, 2)
        return moves

    def generate_en_passants(self, state: State) -> dict:
        if state.possible_en_passant_target == -1:
            return {}
        moves = {}
        if state.active_color:
            k = 4
        else:
            k = 3
        l = state.possible_en_passant_target
        dl = [-1, 1]
        directions = [[self.Direction.NORTHWEST, self.Direction.NORTHEAST],
                      [self.Direction.SOUTHWEST, self.Direction.SOUTHEAST]]
        for m in range(2):
            if l + dl[m] < 0 or l + dl[m] >= 8:
                continue
            if state.active_color and state.bitboards[BitboardType.BLACK_PAWN] & 1 << (8 * k + (l + dl[m])):
                moves[self.MOVES_PER_LOCATION * (8 * k + (l + dl[m])) + (
                        7 * directions[state.active_color][dl[m] < 0])] = (k, l + dl[m], k + 1, l)
            elif not state.active_color and state.bitboards[BitboardType.WHITE_PAWN] & 1 << (8 * k + (l + dl[m])):
                moves[self.MOVES_PER_LOCATION * (8 * k + (l + dl[m])) + (
                        7 * directions[state.active_color][dl[m] < 0])] = (k, l + dl[m], k - 1, l)
        return moves

    def generate_king_moves(self, state: State) -> dict:
        if state.active_color:
            kings = state.bitboards[BitboardType.BLACK_KING]
        else:
            kings = state.bitboards[BitboardType.WHITE_KING]
        moves = {}
        dr_to_direction = {
            (-1, 0): self.Direction.NORTH,
            (-1, 1): self.Direction.NORTHEAST,
            (0, 1): self.Direction.EAST,
            (1, 1): self.Direction.SOUTHEAST,
            (1, 0): self.Direction.SOUTH,
            (1, -1): self.Direction.SOUTHWEST,
            (0, -1): self.Direction.WEST,
            (-1, -1): self.Direction.NORTHWEST
        }
        while kings > 0:
            k = int(math.log2(kings & -kings))
            i = k // 8
            j = k % 8
            if state.active_color:
                attack_set = self.king_attack_sets[k] & ~state.bitboards[BitboardType.BLACK]
            else:
                attack_set = self.king_attack_sets[k] & ~state.bitboards[BitboardType.WHITE]
            while attack_set > 0:
                n = int(math.log2(attack_set & -attack_set))
                l = n // 8
                m = n % 8
                moves[self.MOVES_PER_LOCATION * k + (7 * dr_to_direction[(l - i, m - j)])] = (i, j, l, m)
                attack_set -= 1 << n
            kings -= 1 << k
        return moves

    def generate_knight_moves(self, state: State) -> dict:
        if state.active_color:
            knights = state.bitboards[BitboardType.BLACK_KNIGHT]
        else:
            knights = state.bitboards[BitboardType.WHITE_KNIGHT]
        moves = {}
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
            if state.active_color:
                attack_set = self.knight_attack_sets[k] & ~state.bitboards[BitboardType.BLACK]
            else:
                attack_set = self.knight_attack_sets[k] & ~state.bitboards[BitboardType.WHITE]
            while attack_set > 0:
                n = int(math.log2(attack_set & -attack_set))
                l = n // 8
                m = n % 8
                moves[self.MOVES_PER_LOCATION * k + (self.KNIGHT_MOVE_INDEX + dr_to_index[(l - i, m - j)])] = (
                    i, j, l, m)
                attack_set -= 1 << n
            knights -= 1 << k
        return moves

    def generate_possible_masked_blockers(self, masked_blockers: int, k: attack_sets, int, magic_bitboard_type: MagicBitboardType, l: int) -> None:
        if k == 64:
            self.magic_bitboards[masked_blockers] = self.get_sliding_piece_attack_set(masked_blockers, l, magic_bitboard_type)
            return
        if not masked_blockers & 1 << k:
            self.generate_possible_masked_blockers(masked_blockers, k + 1, attack_sets, l,
                                                   magic_bitboard_type)
            return
        masked_blockers -= 1 << k
        self.generate_possible_masked_blockers(masked_blockers, k + 1, attack_sets, l,
                                               magic_bitboard_type)
        masked_blockers += 1 << k
        self.generate_possible_masked_blockers(masked_blockers, k + 1, attack_sets, l,
                                               magic_bitboard_type)

    def generate_pawn_captures(self, state: State) -> dict:
        if state.active_color:
            pawns = state.bitboards[BitboardType.BLACK_PAWN]
        else:
            pawns = state.bitboards[BitboardType.WHITE_PAWN]
        moves = {}
        directions = [[self.Direction.NORTHWEST, self.Direction.NORTHEAST],
                      [self.Direction.SOUTHWEST, self.Direction.SOUTHEAST]]
        while pawns > 0:
            k = int(math.log2(pawns & -pawns))
            i = k // 8
            j = k % 8
            if state.active_color:
                attack_set = self.pawn_attack_sets[self.PawnAttackSetType.BLACK][k] & state.bitboards[
                    BitboardType.WHITE]
            else:
                attack_set = self.pawn_attack_sets[self.PawnAttackSetType.WHITE][k] & state.bitboards[
                    BitboardType.BLACK]
            promotion_i = [1, 6]
            dj_to_capture_underpromotion_type = {
                -1: self.CaptureUnderpromotionType.LEFT,
                1: self.CaptureUnderpromotionType.RIGHT
            }
            while attack_set > 0:
                n = int(math.log2(attack_set & -attack_set))
                l = n // 8
                m = n % 8
                moves[self.MOVES_PER_LOCATION * k + (7 * directions[state.active_color][m > j])] = (i, j, l, m)
                if i == promotion_i[state.active_color]:
                    for l in range(3):
                        moves[self.MOVES_PER_LOCATION * k + (
                                self.UNDERPROMOTION_INDEX + 3 * dj_to_capture_underpromotion_type[(m - j)] + l)] = (
                            i, j, l, m)
                attack_set -= 1 << n
            pawns -= 1 << k
        return moves

    def generate_pawn_one_forward_moves(self, state: State) -> dict:
        if state.active_color:
            pawns = state.bitboards[BitboardType.EMPTY] >> 8 & state.bitboards[BitboardType.BLACK_PAWN]
        else:
            pawns = state.bitboards[BitboardType.EMPTY] << 8 & state.bitboards[BitboardType.WHITE_PAWN]
        moves = {}
        di = [-1, 1]
        directions = [self.Direction.NORTH, self.Direction.SOUTH]
        promotion_i = [1, 6]
        while pawns > 0:
            k = int(math.log2(pawns & -pawns))
            i = k // 8
            j = k % 8
            moves[self.MOVES_PER_LOCATION * k + (7 * directions[state.active_color])] = (
                i, j, i + di[state.active_color], j)
            if i == promotion_i[state.active_color]:
                for l in range(3):
                    moves[self.MOVES_PER_LOCATION * k + (self.UNDERPROMOTION_INDEX + l)] = (
                        i, j, i + di[state.active_color], j)
            pawns -= 1 << k
        return moves

    def generate_pawn_two_forward_moves(self, state: State) -> dict:
        if state.active_color:
            pawns = (state.bitboards[BitboardType.EMPTY] >> 8) \
                    & (state.bitboards[BitboardType.EMPTY] >> 16) \
                    & state.bitboards[BitboardType.BLACK_PAWN] \
                    & int(("00000000" +
                           "11111111" +
                           "00000000" +
                           "00000000" +
                           "00000000" +
                           "00000000" +
                           "00000000" +
                           "00000000"
                           [::-1]), 2)
        else:
            pawns = (state.bitboards[BitboardType.EMPTY] << 8) \
                    & (state.bitboards[BitboardType.EMPTY] << 16) \
                    & state.bitboards[BitboardType.WHITE_PAWN] \
                    & int(("00000000" +
                           "00000000" +
                           "00000000" +
                           "00000000" +
                           "00000000" +
                           "00000000" +
                           "11111111" +
                           "00000000")
                          [::-1], 2)
        moves = {}
        di = [-1, 1]
        directions = [self.Direction.NORTH, self.Direction.SOUTH]
        while pawns > 0:
            k = int(math.log2(pawns & -pawns))
            i = k // 8
            j = k % 8
            moves[self.MOVES_PER_LOCATION * k + (7 * directions[state.active_color] + 1)] = (
                i, j, i + 2 * di[state.active_color], j)
            pawns -= 1 << k
        return moves

    def get_sliding_piece_attack_set(self, masked_blockers: int, k: int, magic_bitboard_type: MagicBitboardType) -> int:
        if magic_bitboard_type == self.MagicBitboardType.COLUMN:
            north_masked_blockers = masked_blockers & self.rays[self.Direction.NORTH][k]
            east_masked_blockers = masked_blockers & self.rays[self.Direction.EAST][k]
            south_masked_blockers = masked_blockers & self.rays[self.Direction.SOUTH][k]
            west_masked_blockers = masked_blockers & self.rays[self.Direction.WEST][k]
            if north_masked_blockers == 0:
                excluded_north_ray = 0
            else:
                excluded_north_ray = self.rays[self.Direction.NORTH][int(math.log2(north_masked_blockers))]
            if east_masked_blockers == 0:
                excluded_east_ray = 0
            else:
                excluded_east_ray = self.rays[self.Direction.EAST][int(math.log2(east_masked_blockers & -east_masked_blockers))]
            if south_masked_blockers == 0:
                excluded_south_ray = 0
            else:
                excluded_south_ray = self.rays[self.Direction.SOUTH][int(math.log2(south_masked_blockers & -south_masked_blockers))]
            if west_masked_blockers == 0:
                excluded_west_ray = 0
            else:
                excluded_west_ray = self.rays[self.Direction.WEST][int(math.log2(west_masked_blockers))]
            return self.rays[self.Direction.NORTH][k] & ~excluded_north_ray | \
                   self.rays[self.Direction.EAST][k] & ~excluded_east_ray | \
                   self.rays[self.Direction.SOUTH][k] & ~excluded_south_ray | \
                   self.rays[self.Direction.WEST][k] & ~excluded_west_ray
        else:
            northeast_masked_blockers = masked_blockers & self.rays[self.Direction.NORTHEAST][k]
            southeast_masked_blockers = masked_blockers & self.rays[self.Direction.SOUTHEAST][k]
            southwest_masked_blockers = masked_blockers & self.rays[self.Direction.SOUTHWEST][k]
            northwest_masked_blockers = masked_blockers & self.rays[self.Direction.NORTHWEST][k]
            if northeast_masked_blockers == 0:
                excluded_northeast_ray = 0
            else:
                excluded_northeast_ray = self.rays[self.Direction.NORTHEAST][
                int(math.log2(northeast_masked_blockers))]
            if southeast_masked_blockers == 0:
                excluded_southeast_ray = 0
            else:
                excluded_southeast_ray = self.rays[self.Direction.SOUTHEAST][
                int(math.log2(southeast_masked_blockers & -southeast_masked_blockers))]
            if southwest_masked_blockers == 0:
                excluded_southwest_ray = 0
            else:
                excluded_southwest_ray = self.rays[self.Direction.SOUTHWEST][
                int(math.log2(southwest_masked_blockers & -southwest_masked_blockers))]
            if northwest_masked_blockers == 0:
                excluded_northwest_ray = 0
            else:
                excluded_northwest_ray = self.rays[self.Direction.NORTHWEST][
                int(math.log2(northwest_masked_blockers))]
            return self.rays[self.Direction.NORTHEAST][k] & ~excluded_northeast_ray | \
                   self.rays[self.Direction.SOUTHEAST][k] & ~excluded_southeast_ray | \
                   self.rays[self.Direction.SOUTHWEST][k] & ~excluded_southwest_ray | \
                   self.rays[self.Direction.NORTHWEST][k] & ~excluded_northwest_ray

    def initialize_east_attack_sets(self) -> None:
        for i in range(8):
            attack_set = 0
            for j in range(7, -1, -1):
                self.rays[self.Direction.EAST][8 * i + j] = attack_set
                attack_set += 1 << (8 * i + j)

    def initialize_king_attack_sets(self) -> None:
        di = [-1, -1, 0, 1, 1, 1, 0, -1]
        dj = [0, 1, 1, 1, 0, -1, -1, -1]
        for i in range(8):
            for j in range(8):
                for k in range(8):
                    if i + di[k] < 0 or i + di[k] >= 8 or j + dj[k] < 0 or j + dj[k] >= 8:
                        continue
                    self.king_attack_sets[8 * i + j] += 1 << (8 * (i + di[k]) + (j + dj[k]))

    def initialize_knight_attack_sets(self) -> None:
        di = [-2, -1, 1, 2, 2, 1, -1, -2]
        dj = [1, 2, 2, 1, -1, -2, -2, -1]
        for i in range(8):
            for j in range(8):
                for k in range(8):
                    if i + di[k] < 0 or i + di[k] >= 8 or j + dj[k] < 0 or j + dj[k] >= 8:
                        continue
                    self.knight_attack_sets[8 * i + j] += 1 << (8 * (i + di[k]) + (j + dj[k]))

    def initialize_magic_bitboards(self) -> None:
        for k in range(64):
            masked_blockers = self.rays[self.Direction.NORTH][k] \
                              | self.rays[self.Direction.EAST][k] \
                              | self.rays[self.Direction.SOUTH][k] \
                              | self.rays[self.Direction.WEST][k]
            self.generate_possible_masked_blockers(masked_blockers, 0, self.MagicBitboardType.COLUMN, k)
            masked_blockers = self.rays[self.Direction.NORTHEAST][k] \
                              | self.rays[self.Direction.SOUTHEAST][k] \
                              | self.rays[self.Direction.SOUTHWEST][k] \
                              | self.rays[self.Direction.NORTHWEST][k]
            attack_sets = {}
            self.generate_possible_masked_blockers(masked_blockers, 0, self.MagicBitboardType.DIAGONAL, k)

    def initialize_north_attack_sets(self) -> None:
        for j in range(8):
            attack_set = 0
            for i in range(8):
                self.rays[self.Direction.NORTH][8 * i + j] = attack_set
                attack_set += 1 << (8 * i + j)

    def initialize_northeast_attack_sets(self) -> None:
        for j in range(8):
            attack_set = 0
            i = 0
            k = 0
            while i + k < 8 and j - k >= 0:
                self.rays[self.Direction.NORTHEAST][8 * (i + k) + (j - k)] = attack_set
                attack_set += 1 << (8 * (i + k) + (j - k))
                k += 1
        for i in range(1, 8):
            attack_set = 0
            j = 7
            k = 0
            while i + k < 8 and j - k >= 0:
                self.rays[self.Direction.NORTHEAST][8 * (i + k) + (j - k)] = attack_set
                attack_set += 1 << (8 * (i + k) + (j - k))
                k += 1

    def initialize_northwest_attack_sets(self) -> None:
        for j in range(8):
            attack_set = 0
            i = 0
            k = 0
            while i + k < 8 and j + k < 8:
                self.rays[self.Direction.NORTHWEST][8 * (i + k) + (j + k)] = attack_set
                attack_set += 1 << (8 * (i + k) + (j + k))
                k += 1
        for i in range(1, 8):
            attack_set = 0
            j = 0
            k = 0
            while i + k < 8 and j + k < 8:
                self.rays[self.Direction.NORTHWEST][8 * (i + k) + (j + k)] = attack_set
                attack_set += 1 << (8 * (i + k) + (j + k))
                k += 1

    def initialize_pawn_attack_sets(self) -> None:
        dj = [-1, 1]
        for i in range(1, 8):
            for j in range(8):
                for k in range(2):
                    if j + dj[k] < 0 or j + dj[k] >= 8:
                        continue
                    self.pawn_attack_sets[self.PawnAttackSetType.WHITE][8 * i + j] += 1 << (8 * (i - 1) + (j + dj[k]))
        for i in range(7):
            for j in range(8):
                for k in range(2):
                    if j + dj[k] < 0 or j + dj[k] >= 8:
                        continue
                    self.pawn_attack_sets[self.PawnAttackSetType.BLACK][8 * i + j] += 1 << (8 * (i + 1) + (j + dj[k]))

    def initialize_sliding_piece_attack_sets(self) -> None:
        self.initialize_north_attack_sets()
        self.initialize_northeast_attack_sets()
        self.initialize_east_attack_sets()
        self.initialize_southeast_attack_sets()
        self.initialize_south_attack_sets()
        self.initialize_southwest_attack_sets()
        self.initialize_west_attack_sets()
        self.initialize_northwest_attack_sets()

    def initialize_south_attack_sets(self) -> None:
        for j in range(8):
            attack_set = 0
            for i in range(7, -1, -1):
                self.rays[self.Direction.SOUTH][8 * i + j] = attack_set
                attack_set += 1 << (8 * i + j)

    def initialize_southeast_attack_sets(self) -> None:
        for j in range(8):
            attack_set = 0
            i = 7
            k = 0
            while i - k >= 0 and j - k >= 0:
                self.rays[self.Direction.SOUTHEAST][8 * (i - k) + (j - k)] = attack_set
                attack_set += 1 << (8 * (i - k) + (j - k))
                k += 1
        for i in range(7):
            attack_set = 0
            j = 7
            k = 0
            while i - k >= 0 and j - k >= 0:
                self.rays[self.Direction.SOUTHEAST][8 * (i - k) + (j - k)] = attack_set
                attack_set += 1 << (8 * (i - k) + (j - k))
                k += 1

    def initialize_southwest_attack_sets(self) -> None:
        for j in range(8):
            attack_set = 0
            i = 7
            k = 0
            while i - k >= 0 and j + k < 8:
                self.rays[self.Direction.SOUTHWEST][8 * (i - k) + (j + k)] = attack_set
                attack_set += 1 << (8 * (i - k) + (j + k))
                k += 1
        for i in range(7):
            attack_set = 0
            j = 0
            k = 0
            while i - k >= 0 and j + k < 8:
                self.rays[self.Direction.SOUTHWEST][8 * (i - k) + (j + k)] = attack_set
                attack_set += 1 << (8 * (i - k) + (j + k))
                k += 1

    def initialize_west_attack_sets(self) -> None:
        for i in range(8):
            attack_set = 0
            for j in range(8):
                self.rays[self.Direction.WEST][8 * i + j] = attack_set
                attack_set += 1 << (8 * i + j)

    def print(self, b):
        for i in range(8):
            for j in range(8):
                if b & 1 << (8 * i + j):
                    print('1', end='')
                else:
                    print('0', end='')
            print()
        print()
