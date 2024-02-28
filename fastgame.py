from numba import njit
import numpy as np
from time import time

@njit('int_(uint64)')
def popcount(b):
    b -= (b >> 1) & 0x5555555555555555
    b = (b & 0x3333333333333333) + ((b >> 2) & 0x3333333333333333)
    b = (b + (b >> 4)) & 0x0f0f0f0f0f0f0f0f
    return (b * 0x0101010101010101) >> 56

# Shitty implementation but numba gives ~20x speedup
@njit('uint64(uint64, uint64)')
def pext(x, mask):
    result = 0
    x &= mask
    j = 0
    for i in range(64):
        if mask & (1 << i):
            result |= x >> (i-j) & (1 << j)
            j += 1
    return result

@njit('uint64(uint64, uint64)')
def pdep(x, mask):
    result = 0
    for i in range(64):
        if mask & (1 << i):
            result |= (x & 1) << i
            x >>= 1
    return result

def timeit(func, *args):
    start_time = time()
    for _ in range(1000000):
        func.py_func(*args)
    print("Time without numba :", time() - start_time)

    func(*args) #force compilation before timing the loop
    start_time = time()
    for _ in range(1000000):
        func(*args)
    print("Time with numba :", time() - start_time)

#timeit(pext, 0b111010, 0b111001)
#timeit(pdep, 0b1001, 0b111010)


#
# how to find legal moves :
# for each mask representing a row/col/diagonal:
#     pattern_black = pext(black, mask)
#     pattern_white = pext(white, mask)
#     legal = TABLE[pattern_black][pattern_white]
#     legal_moves |= pdep(legal, mask)
#
# What will be the size of TABLE ? well, 2**8 * 2**8 = 2**16 = 65536
# 65536 * 8 = 524288 bytes = ~500KB
# One could gain some space by encoding the patterns in ternary, 
# but I want to gain performance rather than a few KBs.
#
# Notice that even for a pattern which has less than 8 squares, for example
# a 5-diagonal, we can still use the same table, and the 3 missing squares
# will be considered as empty. It might generate some wrong moves outside of the 
# 5 squares, but we will simply ignore them.
#

#
# how to find discs to flip given a move :
# precompute a table which associates to any move (m between 0 and 63)
# 4 masks representing the row, col, diag1, diag2 containing m (masks in the form
# of lists of indices). Then, use pext to get patterns and use a precomputed 
# table to find the discs to flip in all 4 directions.
#

@njit('uint64[:,:]()')
def gen_table():
    '''
    Generate the table of legal moves for each pattern of black and white discs.
    Also stores it in a file to avoid regenerating it each time.
    '''
    #always assume black is the one playing
    TABLE = np.zeros((2**8, 2**8), dtype=np.uint64)
    for pattern_black in range(2**8):
        for pattern_white in range(2**8):
            empty = 0xFF ^ (pattern_black | pattern_white)
            legal = 0
            if pattern_black & pattern_white: #invalid pattern
                continue
            B = pattern_black
            W = pattern_white 
            while B:
                B = (B >> 1) & W #B now contains white discs on the right of a black disc.
                legal |= (B >> 1) & empty
            B = pattern_black
            while B:
                B = (B << 1) & W
                legal |= (B << 1) & empty
            TABLE[pattern_black][pattern_white] = legal
    return TABLE

TABLE = np.load('data/move_gen_table.npy')

DIRECTIONS = [0] * 38
for i in range(8): # rows
    DIRECTIONS[i] = 255 << (i * 8) 
for i in range(8): # cols
    DIRECTIONS[i + 8] = 0x0101010101010101 << i
diag = 0x8040201008040201
for i in range(6): # upper diagonals from 8-diag to 3-diag
    DIRECTIONS[i + 16] = diag
    diag = diag >> 1 & 0x7F7F7F7F7F7F7F7F
diag = 0x40201008040201 << 1 
for i in range(5): # lower diagonals from 7-diag to 3-diag
    DIRECTIONS[i + 22] = diag
    diag = diag << 1 & 0xFEFEFEFEFEFEFEFE
diag = 0x0102040810204080
for i in range(6): # lower diagonals from 8-diag to 3-diag in the other direction
    DIRECTIONS[i + 27] = diag
    diag = diag >> 1 & 0x7F7F7F7F7F7F7F7F
diag = 0x0102040810204000 << 1
for i in range(5): # upper diagonals from 7-diag to 3-diag in the other direction
    DIRECTIONS[i + 33] = diag
    diag = diag << 1 & 0xFEFEFEFEFEFEFEFE

npdirections = np.array(DIRECTIONS, dtype=np.uint64)

def print_bitboard(x):
    if isinstance(x, np.uint64):
        x = int(x)
    for i in range(8):
        print(bin(x >> ((7-i) * 8) & 0xFF)[2:].zfill(8))



class BBPosition:
    '''
    A class to represent the board and the game state using 64-bit integers.
    Side is 1 for black, -1 for white.
    '''
    def __init__(self, black=34628173824, white=68853694464, side=1):
        '''
        Default values are the starting position.
        '''
        self.black = np.uint64(black)
        self.white = np.uint64(white)
        self.side = side

    def gen_moves(self):
        '''
        Returns a list of tuples of legal moves and the discs flipped for each of them.
        A move is an integer between 0 and 63, the discs to flip are represented as a bitboard.
        '''
        legal_bb = fast_gen_moves(self.black, self.white, self.side, npdirections, TABLE)
        legal = np.array([i for i in range(64) if legal_bb >> i & 1], dtype=np.int8)
        return [(m, get_flips(self.black, self.white, self.side, m)) for m in legal]
    

    def can_play(self):
        return fast_gen_moves(self.black, self.white, self.side, npdirections, TABLE) != 0


    def is_legal_move(self, m):
        '''
        Returns True if the move is legal, False otherwise.
        '''
        if (fast_gen_moves(self.black, self.white, self.side, npdirections, TABLE) >> m) & 1:
            return True, np.uint64(get_flips(self.black, self.white, self.side, m))
        else: 
            return False, np.uint64(0)
    
    def make_move(self, m, flip):
        '''
        Returns a new position after the move is made.
        '''
        flip = np.uint64(flip)
        newp = BBPosition(self.black, self.white, self.side)
        if self.side == 1:
            newp.black |= (np.uint64(1) << np.uint(m)) | flip
            newp.white ^= flip
        else:
            newp.white |= (np.uint64(1) << np.uint(m)) | flip
            newp.black ^= flip
        newp.side = -self.side
        if not fast_gen_moves(newp.black, newp.white, newp.side, npdirections, TABLE):
            newp.side = self.side # pass if no move available
        return newp

    def undo_move(self, m, flip):
        '''
        Returns the position before the move was made.
        '''
        flip = np.uint64(flip)
        newp = BBPosition(self.black, self.white, self.side)
        newp.side = 1 if self.black & np.uint64(1) << np.uint64(m) else -1
        if newp.side == 1:
            newp.black ^= (np.uint64(1) << np.uint64(m)) | flip
            newp.white ^= flip
        else:
            newp.white ^= (np.uint64(1) << np.uint64(m)) | flip
            newp.black ^= flip
        return newp
    
    def result(self):
        '''
        Returns the result of the game, in the form black discs - white discs.
        Any empty squares are given to the winning player.
        '''
        black = popcount(self.black)
        white = popcount(self.white)
        return black - white + (64 - black - white) * np.sign(black - white)
        
        
    def __repr__(self):
        s = '  a b c d e f g h\n'
        for i in range(64):
            if i % 8 == 0:
                s += '12345678'[i // 8] + ' '
            if (self.black >> np.uint(i)) & np.uint64(1):
                s += 'X '
            elif (self.white >> np.uint(i)) & np.uint64(1):
                s += 'O '
            else:
                s += '. '
            if i % 8 == 7:
                s += '\n'
        s += 'Side to play : ' + ('X' if self.side == 1 else 'O') + '\n'
        return s

@njit('uint64(uint64, uint64, int64, uint64[:], uint64[:,:])')
def fast_gen_moves(black, white, side, npdirections, table):
    if side == -1:
        black, white = white, black
    legal_moves = np.uint64(0)
    for mask in npdirections:
        pattern_black = pext(black, mask)
        pattern_white = pext(white, mask)
        legal = table[pattern_black][pattern_white]
        legal_moves |= pdep(legal, mask)
    return legal_moves

@njit('uint64(uint64, uint64, int64, int64)')
def get_flips(black, white, side, move):
    '''
    black, white -- bitboards representing black and white discs
    side -- 1 for black, -1 for white
    move -- the move to be played, as an integer between 0 and 63
    
    Returns a bitboard representing the discs to flip.
    '''
    if side == -1:
        black, white = white, black
    m = np.uint64(1) << np.uint64(move)
    flip = np.uint64(0)
    #mask prevents from flipping discs outside of the board
    for dir, mask in zip(np.array([1, 7, 8, 9], dtype=np.uint64), np.array([0x7F7F7F7F7F7F7F7F, 0x00FEFEFEFEFEFEFE, 0xFFFFFFFFFFFFFFFF, 0x7F7F7F7F7F7F7F7F], dtype=np.uint64)):
        next_disc = m >> dir & mask
        potential_flip = next_disc & white
        next_disc = potential_flip >> dir & mask
        while next_disc & (white ^ potential_flip):
            potential_flip |= next_disc
            next_disc = potential_flip >> dir & mask
        if next_disc & black:
            flip |= potential_flip

    #Do the same thing with << dir instead of >> dir
    for dir, mask in zip(np.array([1, 7, 8, 9], dtype=np.uint64), np.array([0xFEFEFEFEFEFEFEFE, 0x7F7F7F7F7F7F7F7F, 0xFFFFFFFFFFFFFFFF, 0xFEFEFEFEFEFEFEFE], dtype=np.uint64)):
        next_disc = m << dir & mask
        potential_flip = next_disc & white
        next_disc = potential_flip << dir & mask
        while next_disc & (white ^ potential_flip):
            potential_flip |= next_disc
            next_disc = potential_flip << dir & mask
        if next_disc & black:
            flip |= potential_flip
    return flip
        
        
#TABLE = gen_table()
#np.save('data/move_gen_table.npy', TABLE)

'''
position = BBPosition()
move_history = []
while True:
    cmd = input('>')
    if cmd.split()[0] == 'move':
        col = ord(cmd.split()[1][0]) - ord('a') + 1
        row = int(cmd.split()[1][1])
        m = (row - 1) * 8 + col - 1
        print(m)
        legal, flip = position.is_legal_move(m)
        if legal:
            position = position.make_move(m, flip)
            print(position)
            move_history.append((m, flip))
        else:
            print("Illegal move")
    elif cmd == 'undo':
        if move_history:
            m, flip = move_history.pop()
            position = position.undo_move(m, flip)
            print(position)
        else:
            print("No move to undo")
'''

'''
from game import Position, start_pos
start_time = time()
pos = Position(start_pos, 'X')
start_time = time()
for _ in range(10000):
    pos.gen_moves()
print("Time with game.py :", time() - start_time)

pos = BBPosition()
print(pos.gen_moves())
start_time = time()
for _ in range(10000):
    pos.gen_moves()
print("Time with BBPosition :", time() - start_time)
'''
