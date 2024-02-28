from collections import namedtuple

#
# Board representation : string of 100 characters (10x10 array)
# First/last row and column are used to easily check boundaries
# The starting position is encoded as follows:
#  "          "
#  " ........ "
#  " ........ "
#  " ........ "
#  " ...OX... "
#  " ...XO... "
#  " ........ "
#  " ........ "
#  " ........ "
#  "          "
# 


###############################################################################
# Game Logic
###############################################################################

Move = namedtuple('Move', 'row col flip')

class Position():
    def __init__(self, board, side):
        self.board = board
        self.side = side
        self.opponent = 'O' if side == 'X' else 'X'

    def gen_moves(self):
        moves = []
        for row in range(1, 9):
            for col in range(1, 9):
                legal, flip = self.is_legal_move(row, col)
                if legal:
                    moves.append(Move(row, col, flip))
        return moves
    
    def is_legal_move(self, row, col):
        '''
        Check if the move is legal. Returns True, [discs to flip] if the move is legal. 
        Otherwise returns False, []
        '''
        if self.board[row * 10 + col] != '.':
            return False, []
        flipped = []
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue
                _, discs = self.is_legal_direction(row, col, dr, dc)
                flipped.extend(discs)
        return len(flipped) > 0, flipped
    
    def is_legal_direction(self, row, col, dr, dc):
        '''
        Check if the move is legal in the direction (dr, dc).
        Returns False, [] if the move is illegal, otherwise returns True, [discs to flip]
        '''
        r = row + dr
        c = col + dc
        flipped = []
        while self.board[r * 10 + c] == self.opponent:
            flipped.append((r, c))
            r += dr
            c += dc
            if self.board[r * 10 + c] == self.side:
                return True, flipped
        return False, []
    
    def make_move(self, move):
        '''
        Returns a new position after the move is made.
        '''
        put = lambda board, i, s: board[:i] + s + board[i + 1:]

        new_board = put(self.board, move.row * 10 + move.col, self.side)
        for r, c in move.flip:
            new_board = put(new_board, r * 10 + c, self.side)

        #Usually we now switch sides, but we need to check whether the opponent can move
        new_position = Position(new_board, self.opponent)
        if len(new_position.gen_moves()) > 0:
            return new_position
        else:
            return Position(new_board, self.side) 
        
    def undo_move(self, move):
        '''
        Returns the position before the move was made.
        '''
        #Helper function to update the board
        put = lambda board, i, s: board[:i] + s + board[i + 1:]

        #Figure out who played and create the Position object
        prev_side = self.board[move.row * 10 + move.col]
        prev_position = Position(self.board, prev_side)

        #Remove disc placed then revert flipped discs
        prev_position.board = put(self.board, move.row * 10 + move.col, '.')
        for r, c in move.flip:
            prev_position.board = put(prev_position.board, r * 10 + c, prev_position.opponent)
        return prev_position

        
    def result(self):
        '''
        Returns the result of the game, assuming the game is over.
        Returns the difference between black discs and white discs in the score, 
        ie 33:31 -> 2, 31:33 -> -2, 64:0 -> 64, 34:28 -> 8 because the empty squares are given to the winning player.
        '''
        black_score = sum([1 for i in range(100) if self.board[i] == 'X'])
        white_score = sum([1 for i in range(100) if self.board[i] == 'O'])
        if black_score > white_score:
            black_score += (64 - black_score - white_score)
        elif white_score > black_score:
            white_score += (64 - black_score - white_score)
        else:
            return 0
        return black_score - white_score
        
    def __repr__(self):
        s = "  a b c d e f g h"
        for row in range(1, 9):
            s += "\n" + str(row) + " "
            for col in range(1, 9):
                s += self.board[row * 10 + col] + " "
        s += "\n  a b c d e f g h \n"
        s += "Side to play :" + self.side
        return s

    def __str__(self):
        return repr(self)
        

start_pos = ( "          "
              " ........ "
              " ........ "
              " ........ "
              " ...OX... "
              " ...XO... "
              " ........ "
              " ........ "
              " ........ "
              "          ")
