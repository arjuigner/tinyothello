from collections import namedtuple
import random
import time

import eval_fct
from game import Move, Position, start_pos
from fastgame import BBPosition

###############################################################################
# Square value table, used to calculate the (dummy) evaluation function 
###############################################################################

square_values = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0,
                0, 100, -8, 8, 6, 6, 8, -8, 100, 0,
                0, -8, -24, -4, -3, -3, -4, -24, -8, 0,
                0, 8, -4, 7, 4, 4, 7, -4, 8, 0,
                0, 6, -3, 4, 7, 7, 4, -3, 6, 0,
                0, 6, -3, 4, 7, 7, 4, -3, 6, 0,
                0, 8, -4, 7, 4, 4, 7, -4, 8, 0,
                0,-8, -24, -4, -3, -3, -4, -24, -8, 0,
                0, 100, -8, 8, 6, 6, 8, -8, 100, 0,
                0,0,0,0,0,0,0,0,0,0]
MAX_SCORE = sum([abs(x) for x in square_values])
MIN_SCORE = -MAX_SCORE
def dummy_evaluate(position):
    black_score = sum([square_values[i] for i in range(100) if position.board[i] == 'X'])
    white_score = sum([square_values[i] for i in range(100) if position.board[i] == 'O'])
    return (black_score - white_score) * (1 if position.side == 'X' else -1)

###############################################################################
# AI
###############################################################################

MAX_SCORE = 100
MIN_SCORE = -MAX_SCORE

TTKey = namedtuple('TTKey', 'black white side')
TTEntry = namedtuple('TTEntry', 'value depth flag')

class Searcher():
    '''
    A simple NegaC* searcher for Othello.
    '''
    def __init__(self):
        self.nodes = 0 #Number of nodes visited
        self.tt = {} #Transposition table

    def search(self, position, depth, lower_bound=MIN_SCORE, upper_bound=MAX_SCORE, eps=1):
        self.nodes = 0
        while lower_bound < upper_bound - eps:
            new_bound = (lower_bound + upper_bound) / 2
            R = self.failsoft_alphabeta(position, new_bound, new_bound+1, depth)
            if R >= new_bound + 1:
                lower_bound = R
            elif R <= new_bound:
                upper_bound = R
            else: 
                return R
        return lower_bound
    
    def failsoft_alphabeta(self, position, alpha, beta, depth):
        '''
        Failsoft alpha-beta search with a fixed depth.
        How to interpret the return value R:
        if R >= beta : cutoff, meaning you can get a score of at least R, which is above beta anyways, 
            so the opponent won't allow this position.
        if R <= alpha : you can get a score of at most R, which is below alpha, so you won't allow this position if alpha
            is the best score you can get so far. However, if the function was only called to know whether the score is 
            below or above alpha, then you get an even better upper bound=R in this case. Note : it is not an exact value because 
            the opponent might have pruned some moves that would have given you an even worse score since they assumed you could get 
            at least alpha anyways.
        if alpha < R < beta : the best score you can get from this position is R, which is between alpha and beta, 
            so it is the "exact" value of the position (given the evaluation function).
        '''
        self.nodes += 1
        #Use the transposition table if possible
        key = TTKey(position.black, position.white, position.side)
        if key in self.tt and self.tt[key].depth >= depth:
            entry = self.tt[key]
            if entry.flag == 'exact':
                return entry.value
            if entry.flag == 'lowerbound':
                alpha = max(alpha, entry.value)
            if entry.flag == 'upperbound':
                beta = min(beta, entry.value)
            if alpha >= beta:
                return entry.value

        best_score = min(-10000000, alpha)
        if not position.can_play():
            return position.result() * (1 if position.side == 'X' else -1)
        elif depth == 0:
            return eval_fct.evaluate(position)
        
        side = position.side
        score = -10000000
        for new_position in self.get_ordered_children(position):
            if side != new_position.side:
                score = -self.failsoft_alphabeta(new_position, -beta, -alpha, depth - 1)
            else: #opponent passed so it is our turn again
                score = self.failsoft_alphabeta(new_position, alpha, beta, depth - 1)
            best_score = max(score, best_score)
            if best_score > alpha:
                alpha = best_score
                if best_score >= beta:
                    break

        #Store our result in the transposition table:
        flag = 'exact' if alpha < best_score < beta else 'lowerbound' if best_score >= beta else 'upperbound'
        self.tt[key] = TTEntry(best_score, depth, flag)

        return best_score
    
    def get_ordered_children(self, position):
        '''
        Order moves using the transposition table information
        TODO : ordering not implemented yet because the evaluation function is not good enough and gives completely different values
        from a move to the next.
        '''
        children = [position.make_move(move, flips) for move, flips in position.gen_moves()]
        children_with_score = []
        for child in children:
            key = TTKey(child.black, child.white, child.side)
            if key in self.tt:
                entry = self.tt[key]
                if child.side == position.side:
                    if entry.flag == 'exact' or entry.flag == 'lowerbound':
                        children_with_score.append((child, entry.value))
                    else:
                        children_with_score.append((child, MIN_SCORE))
                else:
                    if entry.flag == 'exact' or entry.flag == 'upperbound':
                        children_with_score.append((child, -entry.value))
                    else:
                        children_with_score.append((child, MIN_SCORE))
            else:
                #TODO : evaluate with depth-3 or somethingl like that
                children_with_score.append((child, MIN_SCORE))
        return [c for c, s in 
                sorted(children_with_score, key=lambda x: x[1], reverse=True)]
            
    
def eval_all_moves(position, depth, searcher):
    '''
    Evaluate all moves and return their evaluations in the form 
    list(tuple(int, float)), and an integer representing the total number of nodes visited.
    The ints are the moves, and the floats are the evaluations.
    '''
    total_nodes = 0
    best_score = MIN_SCORE
    res = []
    for move, flips in position.gen_moves():
        new_position = position.make_move(move, flips)
        score = -searcher.search(new_position, depth)
        if new_position.side == position.side:
            score = -score
        res.append((move, score))
        total_nodes += searcher.nodes
    return sorted(res, key=lambda x: x[1], reverse=True), total_nodes


class ComputerPlayer():
    def __init__(self, depth):
        self.searcher = Searcher()
        self.depth = depth
    
    def play(self, position):
        r, nodes = eval_all_moves(position, self.depth, self.searcher)
        _, flip = position.is_legal_move(r[0][0]) # TODO : remove this
        print("Nodes visited : ", nodes)
        return position.make_move(r[0][0], flip)
    
class RandomPlayer(): 
    def play(self, position):
        moves = position.gen_moves()
        if len(moves) == 0:
            return position
        return position.make_move(random.choice(moves))
    
###############################################################################
# Arena : Test two algorithms against each other
###############################################################################

def battle(player1, player2, n_games):
    '''
    Play n_games betwen player1 and player2 (half with player1 starting, half with player2 starting), 
    return (wins, draw, losses, avg_disc_diff) from player1's perspective.
    '''
    wins = 0
    losses = 0
    avg_disc_diff = 0
    for i in range(n_games):
        if i % 2 == 0:
            position = BBPosition()
        else:
            position = BBPosition(side=-1)
        while True:
            if position.side == 1:
                position = player1.play(position)
                if len(position.gen_moves()) == 0:
                    break
            else:
                position = player2.play(position)
                if len(position.gen_moves()) == 0:
                    break
        result = position.result()
        if result > 0:
            wins += 1
        elif result < 0:
            losses += 1
        avg_disc_diff += result
        print(f"Game {i+1} : {wins} - {i + 1 - wins - losses} - {losses} - {avg_disc_diff / (i + 1)}")
    return wins, n_games - wins - losses, losses, avg_disc_diff / n_games
        
def battle_of_depths(d1, d2, n_games):
    '''
    Play n_games between two computer players with depths d1 and d2.
    '''
    print(f"Battle between depth {d1} and depth {d2}.")
    player1 = ComputerPlayer(d1)
    player2 = ComputerPlayer(d2)
    return battle(player1, player2, n_games)

###############################################################################
# Command Line Interface
###############################################################################
        
#Basic interface where the user has the following commands :
# quit : quit
# new : new game
# mode <i> : i=1,2,3 indicate whether you play both sides (i=1), or 'X' (i=2), or 'O' (i=3)
# depth <i> : set the depth of the computer player and of the evaluation engine to i
# move <row><col> : play a move, in the format a1, ..., h8
# undo : undo the last move
# eval : evaluate the current position.
# battle : start a battle between ComputerPlayer and RandomPlayer

if __name__ == "__main__":
    position = BBPosition()
    print(position)
    computer = ComputerPlayer(4)
    computer_sides = []
    searcher = Searcher()
    depth = 3
    #depths used for iterative deepening
    depths = [2, 3]
    move_history = []
    while True:
        cmd = input("Enter command : ")
        if cmd == 'quit':
            break
        if cmd == 'new':
            position = BBPosition()
            print(position)
            move_history = []
        if cmd.split()[0] == 'mode':
            computer_sides = {'1': [], '2': ['O'], '3': ['X']}[cmd.split()[1]]

        if cmd.split()[0] == 'depth':
            depth = int(cmd.split()[1])
            computer.depth = depth
            depths = list(range(0, depth+1, 2))
            #depths = [depth]

        
        if cmd.split()[0] == 'move':
            col = ord(cmd.split()[1][0]) - ord('a')
            row = int(cmd.split()[1][1]) - 1
            legal, flip = position.is_legal_move(row*8+col)
            print(type(flip))
            if legal:
                position = position.make_move(row*8+col, flip)
                print(position)
                if position.side in computer_sides:
                    print("It is now the computer's turn.")
                    position = computer.play(position)
                    print(position)
                move_history.append((row*8+col, flip))
            else:
                print("Illegal move")

        if cmd == 'undo':
            if len(move_history) == 0:
                print("No moves to undo.")
                continue
            position = position.undo_move(*(move_history.pop()))
            print(position)

        if cmd == 'eval':
            for d in depths:
                start_time = time.time()
                evals, nodes = eval_all_moves(position, d, searcher)
                print(f"------- Evaluations at depth {d} -------")
                for move, value in evals:
                    print(f"{'abcdefgh'[move%8]}{move//8+1} : {value}")
                print(f"Nodes visited : {nodes}, time spent : {time.time() - start_time}")
                print("-----------------------------------------")
            

        if cmd == 'battle':
            #battle(computer, RandomPlayer(), 10)
            battle_of_depths(2, 4, 10)

