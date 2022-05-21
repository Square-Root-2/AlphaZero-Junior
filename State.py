from BitboardType import BitboardType


class State(object):

    def __init__(self, fen: str = "rnbqkbnr/pppppppp/8/8/8/8/PPPPPPPP/RNBQKBNR w KQkq - 0 1"):
        self.character_to_bitboard_type = {
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

        k = 0

        # piece placement
        self.bitboards = len(BitboardType) * [0]
        i = 0
        while i < 8:
            j = 0
            while j < 8:
                if fen[k].isalpha():
                    self.bitboards[self.character_to_bitboard_type[fen[k]]] += 1 << (8 * i + j)
                    self.bitboards[BitboardType.BLACK if fen[k].islower() else BitboardType.WHITE] += 1 << (8 * i + j)
                elif fen[k].isnumeric():
                    for _ in range(ord(fen[k]) - ord('0')):
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
        self.white_kingside = False
        self.white_queenside = False
        self.black_kingside = False
        self.black_queenside = False
        while fen[k] != ' ':
            if fen[k] == 'K':
                self.white_kingside = True
            elif fen[k] == 'Q':
                self.white_queenside = True
            elif fen[k] == 'k':
                self.black_kingside = True
            elif fen[k] == 'q':
                self.black_queenside = True
            k += 1

        k += 1

        # possible en passant target
        if fen[k] != '-':
            self.possible_en_passant_target = ord(fen[k]) - ord('a')
        else:
            self.possible_en_passant_target = -1

        k += 2

        # halfmove clock and fullmove number
        l = k + 1
        while fen[l] != ' ':
            l += 1

        self.halfmove_clock = int(fen[k: l])
        self.fullmove_number = int(fen[l + 1:])
