from BitboardType import BitboardType
import MoveIndexing


class State(object):
    CHARACTER_TO_BITBOARD_TYPE = {
        'P': BitboardType.WHITE_PAWN,
        'N': BitboardType.WHITE_KNIGHT,
        'B': BitboardType.WHITE_BISHOP,
        'R': BitboardType.WHITE_ROOK,
        'Q': BitboardType.WHITE_QUEEN,
        'K': BitboardType.WHITE_KING,
        'p': BitboardType.BLACK_PAWN,
        'n': BitboardType.BLACK_KNIGHT,
        'b': BitboardType.BLACK_BISHOP,
        'r': BitboardType.BLACK_ROOK,
        'q': BitboardType.BLACK_QUEEN,
        'k': BitboardType.BLACK_KING
    }

    def __init__(self, fen="rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"):
        k = 0

        # piece placement
        self.bitboards = len(BitboardType) * [0]
        self.mailbox = 64 * ['.']
        i = 0
        while i < 8:
            j = 0
            while j < 8:
                if fen[k].isalpha():
                    self.bitboards[State.CHARACTER_TO_BITBOARD_TYPE[fen[k]]] += 1 << (8 * i + j)
                    self.bitboards[BitboardType.BLACK if fen[k].islower() else BitboardType.WHITE] += 1 << (8 * i + j)
                    self.mailbox[8 * i + j] = fen[k]
                elif fen[k].isnumeric():
                    for l in range(ord(fen[k]) - ord('0')):
                        self.bitboards[BitboardType.EMPTY] += 1 << (8 * i + j)
                        j += 1
                    j -= 1
                j += 1
                k += 1
            k += 1
            i += 1

        # active color
        if fen[k] == 'w':
            self.active_color = False
        else:
            self.active_color = True

        k += 2

        # castling rights
        self.castling_rights = 0
        while fen[k] != ' ':
            if fen[k] == 'K':
                self.castling_rights += 1 << 0
            elif fen[k] == 'Q':
                self.castling_rights += 1 << 1
            elif fen[k] == 'k':
                self.castling_rights += 1 << 2
            elif fen[k] == 'q':
                self.castling_rights += 1 << 3
            k += 1

        k += 1

        # possible en passant target
        if fen[k] != '-':
            self.possible_en_passant_target = ord(fen[k]) - ord('a')
        else:
            self.possible_en_passant_target = -1

        k += 2

        # halfmove clock
        l = k + 1
        while fen[l] != ' ':
            l += 1

        self.halfmove_clock = int(fen[k: l])

        self.hash_code = (self.mailbox, self.active_color, self.castling_rights, self.possible_en_passant_target, self.halfmove_clock)

    def clone(self):
        state = State()
        state.set_state(self.hash_code)
        return state

    def get_key(self):
        return tuple(self.mailbox), self.active_color, self.castling_rights, self.possible_en_passant_target

    def make_move(self, move_index, i, j, k, l):
        castling_row = [7, 0]
        is_pawn_move = self.mailbox[8 * i + j].lower() == 'p'
        is_en_passant = is_pawn_move and j != l and self.mailbox[8 * k + l] == '.'
        is_promotion = is_pawn_move and (k == 0 or k == 7)
        is_king_move = self.mailbox[8 * i + j].lower() == 'k'
        is_kingside_castling = is_king_move and l - j == 2
        is_queenside_castling = is_king_move and l - j == -2
        is_kingside_rook_move = self.mailbox[8 * i + j].lower() == 'r' and i == castling_row[self.active_color] and j == 7
        is_queenside_rook_move = self.mailbox[8 * i + j].lower() == 'r' and i == castling_row[self.active_color] and j == 0
        is_kingside_rook_capture = self.mailbox[8 * k + l].lower() == 'r' and k == castling_row[not self.active_color] and l == 7
        is_queenside_rook_capture = self.mailbox[8 * k + l].lower() == 'r' and k == castling_row[not self.active_color] and l == 0
        is_pawn_two_forward = self.mailbox[8 * i + j].lower() == 'p' and abs(k - i) == 2
        is_capture = self.mailbox[8 * i + j] != '.'

        # update piece placement
        self.set_square(k, l, self.mailbox[8 * i + j])
        self.set_square(i, j, '.')

        if is_en_passant:
            self.set_square(i, l, '.')

        if is_promotion:
            if move_index < MoveIndexing.UNDERPROMOTION_BEGINNING_INDEX:
                if self.active_color:
                    self.set_square(k, l, 'q')
                else:
                    self.set_square(k, l, 'Q')
            elif (move_index - MoveIndexing.UNDERPROMOTION_BEGINNING_INDEX) % 3 == 0:
                if self.active_color:
                    self.set_square(k, l, 'r')
                else:
                    self.set_square(k, l, 'R')
            elif (move_index - MoveIndexing.UNDERPROMOTION_BEGINNING_INDEX) % 3 == 1:
                if self.active_color:
                    self.set_square(k, l, 'b')
                else:
                    self.set_square(k, l, 'B')
            else:
                if self.active_color:
                    self.set_square(k, l, 'n')
                else:
                    self.set_square(k, l, 'N')

        if is_kingside_castling:
            self.set_square(i, 7, '.')
            if self.active_color:
                self.set_square(i, 5, 'r')
            else:
                self.set_square(i, 5, 'R')

        # update castling rights
        if is_queenside_castling:
            self.set_square(i, 0, '.')
            if self.active_color:
                self.set_square(i, 3, 'r')
            else:
                self.set_square(i, 3, 'R')

        if is_king_move:
            if self.active_color:
                self.castling_rights &= ~(1 << 2)
                self.castling_rights &= ~(1 << 3)
            else:
                self.castling_rights &= ~(1 << 0)
                self.castling_rights &= ~(1 << 1)

        if is_kingside_rook_move:
            if self.active_color:
                self.castling_rights &= ~(1 << 2)
            else:
                self.castling_rights &= ~(1 << 0)

        if is_queenside_rook_move:
            if self.active_color:
                self.castling_rights &= ~(1 << 3)
            else:
                self.castling_rights &= ~(1 << 1)

        if is_kingside_rook_capture:
            if self.active_color:
                self.castling_rights &= ~(1 << 0)
            else:
                self.castling_rights &= ~(1 << 2)

        if is_queenside_rook_capture:
            if self.active_color:
                self.castling_rights &= ~(1 << 1)
            else:
                self.castling_rights &= ~(1 << 3)

        # update possible en passant target
        if is_pawn_two_forward:
            self.possible_en_passant_target = j
        else:
            self.possible_en_passant_target = -1

        # update halfmove count
        if is_pawn_move or is_capture:
            self.halfmove_clock = 0
        else:
            self.halfmove_clock += 1

        # update active color
        self.active_color = not self.active_color

        self.hash_code = (self.mailbox, self.active_color, self.castling_rights, self.possible_en_passant_target, self.halfmove_clock)

    def print(self):
        for i in range(8):
            for j in range(8):
                print(self.mailbox[8 * i + j], end='')
            print()
        print()

    def set_square(self, i, j, c):
        if self.mailbox[8 * i + j] != '.':
            self.bitboards[State.CHARACTER_TO_BITBOARD_TYPE[self.mailbox[8 * i + j]]] -= 1 << (8 * i + j)
            if self.mailbox[8 * i + j].islower():
                self.bitboards[BitboardType.BLACK] -= 1 << (8 * i + j)
            else:
                self.bitboards[BitboardType.WHITE] -= 1 << (8 * i + j)
            self.bitboards[BitboardType.EMPTY] += 1 << (8 * i + j)
            self.mailbox[8 * i + j] = '.'

        if c != '.':
            self. bitboards[State.CHARACTER_TO_BITBOARD_TYPE[c]] += 1 << (8 * i + j)
            if c.islower():
                self.bitboards[BitboardType.BLACK] += 1 << (8 * i + j)
            else:
                self.bitboards[BitboardType.WHITE] += 1 << (8 * i + j)
            self.bitboards[BitboardType.EMPTY] -= 1 << (8 * i + j)
            self.mailbox[8 * i + j] = c

    def set_state(self, hash_code):
        self.bitboards = len(BitboardType) * [0]
        for k in range(64):
            if hash_code[0][k] != '.':
                self.bitboards[State.CHARACTER_TO_BITBOARD_TYPE[hash_code[0][k]]] += 1 << k
                if hash_code[0][k].islower():
                    self.bitboards[BitboardType.BLACK] += 1 << k
                else:
                    self.bitboards[BitboardType.WHITE] += 1 << k
            else:
                self.bitboards[BitboardType.EMPTY] += 1 << k

        self.mailbox = list(hash_code[0])
        self.active_color = hash_code[1]
        self.castling_rights = hash_code[2]
        self.possible_en_passant_target = hash_code[3]
        self.halfmove_clock = hash_code[4]
        self.hash_code = hash_code
