from MoveGenerator import MoveGenerator
from State import State

mg = MoveGenerator()
while True:
    fen = input()
    s = State(fen)
    depth = int(input())
    mg.perft_root(s, depth)
