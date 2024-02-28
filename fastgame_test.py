########################################################
# Testing
########################################################

from game import Position, start_pos
from fastgame import BBPosition
import random
from tqdm import tqdm
import numpy as np

def random_position():
    pos = Position(start_pos, 'X')
    for _ in range(random.randint(10, 50)):
        moves = pos.gen_moves()
        if moves == []:
            return pos # game over
        pos = pos.make_move(random.choice(moves))
    return pos

def test_move_generation():
    for iteration in tqdm(range(1000)):
        pos = random_position()
        black, white = np.uint64(0), np.uint64(0)
        for r in range(8):
            for c in range(8):
                if pos.board[(r+1) * 10 + c+1] == 'X':
                    black |= np.uint64(1) << np.uint64(r * 8 + c)
                elif pos.board[(r+1) * 10 + c+1] == 'O':
                    white |= np.uint64(1) << np.uint64(r * 8 + c)
        BBpos = BBPosition(black, white, 1 if pos.side == 'X' else -1)
        BBlegal = BBpos.gen_moves()
        legal = pos.gen_moves()
        assert len(BBlegal) == len(legal)
        for m, flip in BBlegal:
            for move in legal:
                if m == (move.row-1)* 8 + move.col-1:
                    trueflip = np.uint64(0)
                    for r, c in move.flip:
                        trueflip |= np.uint64(1) << np.uint64((r-1) * 8 + c-1)
                    assert flip == trueflip
                    break

if __name__ == "__main__":
    test_move_generation()
    print("All tests passed!")
