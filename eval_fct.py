#from tinyothello import Searcher
from game import Position, Move, start_pos
from fastgame import BBPosition, popcount
from math import tanh
import pickle
import random
import struct
import numpy as np
from tqdm import tqdm


def create_save_dataset(start_year, end_year, stage):
    '''
    start_year -- int, first year to consider
    end_year -- int, last year to consider
    stage -- int between 0 and 14, corresponding to the number of empty squares:
            0 means 0-3 empties, 1 means 4-7 empties, etc.
    ''' 
    prefix = "data/WTH_"
    Xs = [None for _ in range(start_year, end_year + 1)]
    ys = [None for _ in range(start_year, end_year + 1)]
    for year in range(start_year, end_year + 1):
        filename = prefix + str(year) + ".wtb"
        games = read_wthor_file(filename)
        positions = get_positions(games)[stage]
        Xs[year - start_year], ys[year - start_year] = get_labeled_data_patterns(positions)
    X = np.concatenate(Xs, axis=0)
    y = np.concatenate(ys, axis=0)
    np.save(f'X_{stage}_{start_year}-{end_year}.npy', X)
    np.save(f'y_{stage}_{start_year}-{end_year}.npy', y)
    return X, y

def read_wthor_file(filename):
    '''
    filename -- str, name of the file to read

    Returns a list of the games in the file - where a game
    is a list of Move objects.
    '''
    games = []
    with open(filename, 'rb') as f:
        _ = f.read(4)
        n = struct.unpack('l', (f.read(4)))[0]
        print(n, "games to read.")
        _ = f.read(8)
        for i in range(n):
            game = []
            _ = f.read(8)
            moves = struct.unpack('60b', f.read(60))
            for x in moves:
                if x == 0: continue
                if not (0 < x // 10 < 10 and 0 < x % 10 < 10) and x != 0:
                    print("Wtf does this value mean in wthor ?", x)
                move = Move(x // 10, x % 10, [])
                game.append(move)
            games.append(game)
    return games

def get_positions(games, evals=None):
    '''
    games -- list of games, as in list of Move objects, possibly with
    invalid "flip" field.
    evals -- list of lists of evals as returned by edax_eval. If None, the function
    simply returns the positions without corresponding evaluations.

    returns a list L of lists. The list L[i] contains Position objects representing
    the positions of the database with 4*i+1 to 4*i+4 empty squares, exactly as they will be fed
    to the model. If `eval` is not None, also returns a list of lists of the same size with 
    corresponding evaluations.
    '''
    L = [[] for _ in range(15)] #positions
    E = [[] for _ in range(15)] #evals
    for g, game in enumerate(games):
        position = Position(start_pos, 'X')
        for i, move in enumerate(game):
            legal, flip = position.is_legal_move(move.row, move.col)
            if legal:
                position = position.make_move(Move(move.row, move.col, flip))
            else:
                print("Illegal move encountered, wtf?")
                break
            empties = 59 - i

            if evals and evals[g][i]:
                E[empties // 4].append(tanh(evals[g][i] / 8.0))
                L[empties // 4].append(position)
            elif not evals:
                L[empties // 4].append(position)
            else:
                pass #no eval for this position --> end of game, don't need this position.
                
        if len(game) != len(evals[g]) and len(game) != len(evals[g]) + 1:
            print("Game and evals don't have the same length:", len(game), len(evals))
    if evals:
        return L, E
    else:
        return L


def get_labeled_data_patterns(positions, labels=None):
    '''
    positions -- list of Position objects
    labels -- list of labels for the positions. If None, the function will computer
    labels using a Searcher object.

    Returns two numpy arrays : one containing the representations of the positions 
    as arrays of indices (indices of the patterns on the board)
    and the other containing the labels of the positions.
    '''
    if labels is None:
        print("No labels provided, NOT IMPLEMENTED YET.")
        return None

    X = np.zeros((len(positions), 8*N))
    y = np.zeros(len(positions))
    #searcher = Searcher()
    for i, p in enumerate(positions):
        X[i, :] = position2indices(p)
        #if i%1000 == 999:
            #searcher = Searcher()
            # Reset the searcher to make sure its transposition table
            # doesn't blow up. TODO : remove when transposition table
            # size is limited in the searcher implementation.
        #y[i] = labels[i] if labels else label_position(p, 4, searcher)
        y[i] = labels[i]
    return X, y


def get_labeled_data128(positions):
    '''
    positions -- list of Position objects

    Returns two numpy arrays : one containing the representations of the positions 
    as a 128 bits array: the first 64 for the current player's discs, and the 
    64 next for the opponent's discs.
    and the other containing the labels of the positions.
    '''
    X = np.zeros((len(positions), 128), dtype=np.float32)
    y = np.zeros(len(positions), dtype=np.float32)
    for i, position in enumerate(positions):
        repr = np.array([
                        [1 if position.board[y*8+x] == position.side     else 0 for y in range(8) for x in range(8)],
                        [1 if position.board[y*8+x] == position.opponent else 0 for y in range(8) for x in range(8)]
                    ],
                    dtype=np.float32).reshape(128)
        y[i] = label_position(position, 4)
        X[i, :] = repr
    return X, y


def label_position(position, depth, searcher):
    '''
    position -- Position object

    Returns a label for the position. The label is the score of the position as evaluated
    by the Searcher, scaled into the range [-1, 1] (-1 for a large loss, 1 for a large win, 0 for a draw)

    # Question : should we scale by dividing by 64 or by applying a sigmoid function?
    '''
    #Divide by 8 (arbitrary) so that it doesn't go to +-1 to fast, but also not too slowly.
    return tanh(searcher.search(position, depth) / 8)
            

def get_batch(X, y, batch_size):
    '''
    X -- numpy array of shape (N_positions, d), where d is the number of features.
    y -- numpy array of shape (N_positions,), containing the labels
    batch_size -- int, number of positions to return

    Returns a 2-tuple : a numpy array of shape (batch_size, d) containing
    the positions and another array of shape (batch_size,) containing the labels.
    '''
    I = random.choices(list(range(X.shape[0])), k=batch_size)
    Xbatch = X[I, :]
    ybatch = y[I]

    #TODO apply random symmetries if we want to
    return Xbatch, ybatch

def create_dataset_from_edax_evals(start_year, end_year, depth=14):
    '''
    start_year -- int, first year to consider
    end_year -- int, last year to consider
    depth -- depth at which the evaluation were made

    Loads the evals from the files and creates the dataset for each stage.
    Stores those datasets in files with the name f"data/X_s{i}.npy" and f"data/y_s{i}.npy
    where i is the stage of the position (= #empties//4)
    '''
    X = [np.zeros((0, 8*N), dtype=np.int32) for _ in range(15)]
    y = [np.zeros(0, dtype=np.float32) for _ in range(15)]
    evals = None
    for year in (pbar := tqdm(range(start_year, end_year+1))):
        pbar.set_description(f"Adding positions from year {year}")
        with open(f"data/edax{depth}_{year}.dat", "rb") as f:
            evals = pickle.load(f)
        games = read_wthor_file(f"data/WTH_{year}.wtb")
        assert len(games) == len(evals)
        L, E = get_positions(games, evals)

        #for each stage, add positions/labels to the corresponding dataset
        for i, (positions, labels) in enumerate(zip(L, E)):
            X_, y_ = get_labeled_data_patterns(positions, labels)
            X[i] = np.concatenate((X[i], X_), axis=0)
            y[i] = np.concatenate((y[i], y_), axis=0)
    #Save all 15 datasets in their own files.
    for i in range(15):
        np.save(f'data/X_s{i}.npy', X[i])
        np.save(f'data/y_s{i}.npy', y[i])
        print("Shape of X and y for stage i :", X[i].shape, y[i].shape)


####################################################
# Patterns
####################################################

#
# We start with original_patterns, which are the N different type of patterns.
# Then, the function create_symmetries creates the 8 symmetries of each of
# these patterns to generate a 2D numpy array containing the 8N patterns.
#
# If not all patterns have the same size, we pad the indices with the value
# 64 : later, when we represent the boards as np arrays, we add a 65-th component
# (the one with index 64) whose value is 0. This allows to perform the pattern
# extraction on the board using only numpy indexing and without any explicit loops.
#

def create_symmetries(originals):
    patterns = np.ones((8*N, M), dtype=np.int8) * 64
    for i, pattern in enumerate(original_patterns):
        npp = np.array(pattern, dtype=np.int8)
        m = len(pattern)
        x = npp % 8
        y = npp // 8
        patterns[8*i, :m]   = npp
        patterns[8*i+1, :m] = y + 8 * (7 - x)
        patterns[8*i+2, :m] = (7 - x) + 8 * (7 - y)
        patterns[8*i+3, :m] = (7 - y) + 8 * x
        x_mirror = 7 - x
        patterns[8*i+4, :m] = x_mirror + 8 * y
        patterns[8*i+5, :m] = y + 8 * (7 - x_mirror)
        patterns[8*i+6, :m] = (7 - x_mirror) + 8 * (7 - y)
        patterns[8*i+7, :m] = (7 - y) + 8 * x_mirror
    return patterns

original_patterns = [
    [0, 1, 2, 3, 4, 5, 6, 7, 9, 14],   #Edge + X-squares
    [0, 1, 2, 8, 9, 10, 16, 17, 18],   #3x3 block in the corner
    [0, 1, 2, 3, 4, 8, 9, 10, 11, 12], #5x2 block in the corner
    [0, 9, 18, 27, 36, 45, 54, 63],    #8-diagonal
    [1, 10, 19, 28, 37, 46, 55],       #7-diagonal
    [2, 11, 20, 29, 38, 47],           #6-diagonal
    [3, 12, 21, 30, 39],               #5-diagonal
    [8, 9, 10, 11, 12, 13, 14, 15],    #line 2
    [16, 17, 18, 19, 20, 21, 22, 23],  #line 3
    [26, 27, 28, 29, 34, 35, 36, 37],  #2x4 rectangle in the middle
]
N = len(original_patterns)
M = max([len(p) for p in original_patterns]) #longest length of pattern
PATTERNS = create_symmetries(original_patterns)

def position2indices(position, patterns=PATTERNS):
    '''
    position -- a Position object
    patterns -- the 2D numpy array containing the indices for each pattern

    Returns a vector where each component corresponds to the int representation
    of the patterns on the board.
    '''
    board = np.array(
        [1 if position.board[i] == position.side else 
            2 if position.board[i] == position.opponent else 0
         for i in actual_squares()])
    basis = 3 ** np.arange(M)
    return board[patterns] @ basis

def bbposition2indices(bbposition, patterns=PATTERNS):
    '''
    bbposition -- a BBPosition object
    patterns -- the 2D numpy array containing the indices for each pattern

    Returns a vector where each component corresponds to the int representation
    of the patterns on the board.
    '''
    own, opp = (bbposition.black, bbposition.white) \
            if bbposition.side == 1 else (bbposition.white, bbposition.black)

    board = np.array(
            [1 if own >> np.uint(8*r+c) & np.uint64(1) else 
            2 if opp >> np.uint(8*r+c) & np.uint64(1) else 0
         for r in range(8) for c in range(8)] + [0])
    #maybe the above can be rewritten without a loop using numpy operations
    basis = 3 ** np.arange(M)
    return board[patterns] @ basis

def actual_squares():
    for y in range(1, 9):
        for x in range(1, 9):
            yield x + 10*y
    yield 0 #index 0 in a board represented as a 100-string will have
    # value '.', which will effectively add a 0 to the numpy representation
    #board.

##############################################################
# Pattern evaluation
##############################################################
    
class PatternLogisticEvaluator:
    def __init__(self):
        self.weights = [None for _ in range(len(PATTERNS) // 8)]
        for i in range(len(self.weights)):
            self.weights[i] = np.zeros(3 ** len(PATTERNS[8 * i]))
        

    def evaluate(self, position):
        indices = position2indices(position, PATTERNS)
        return 8 * sum([self.weights[i//8][idx] for i, idx in enumerate(indices)])
    
    def evaluate_bb(self, bbposition):
        indices = bbposition2indices(bbposition, PATTERNS)
        return 8 * sum([self.weights[i//8][idx] for i, idx in enumerate(indices)])
    
    def train(self, X, Y, learning_rate=0.01, iterations=10, batch_size=128):
        affected_ind = [None] * batch_size * len(PATTERNS)
        nonzero_grad = [None] * batch_size * len(PATTERNS)

        for iteration in (pbar := tqdm(range(1, iterations+1))):
            loss = 0.0
            batch = get_batch(X, Y, batch_size)
            for i, (x, y) in enumerate(zip(*batch)):
                #Calculate the prediction
                z = sum(self.weights[j//8][idx] for j, idx in enumerate(x))
                pred = tanh(z)
                #Calculate the loss (MSE)
                loss += 0.5 * (pred - y) ** 2
                #Calculate the gradient of the loss with respect to the weights
                #However, since only a few weights are affected by each x, we store
                #the indices of the weights that will be affected, and the value of the gradient
                #at these indices, and then we update the weights.
                for j, idx in enumerate(x):
                    #Store the indices of the weights that will be affected by the gradient
                    #Then store the value of the gradient at these indices
                    affected_ind[i*len(PATTERNS) + j] = (j//8, idx)
                    nonzero_grad[i*len(PATTERNS) + j] = (pred - y) * (1 - pred**2)
            
            if iteration%10 == 0:
                pbar.set_description(f"Loss #{iteration} : {loss / batch_size}")
            
            #Update the weights now that we have the gradients for the whole batch.
            for (j, idx), grad in zip(affected_ind, nonzero_grad):
                self.weights[j][idx] -= learning_rate * grad


    def save(self, filename):
        '''
        Save the weights to a file.
        '''
        np.savez(filename, *self.weights)
    
    def load(self, filename):
        '''
        Load the weights from a file.
        '''
        f = np.load(filename)
        for i, s in enumerate(f.files):
            self.weights[i] = f[s]

def train_all_stages(iterations=2000, lr=0.001, batch_size=256, load_evaluators=False):
    '''
    iterations -- int : number of iterations for the training of each stage
    lr -- learning rate to use for the training
    batch_size -- int : size of the batches to use for training
    load_eval -- bool : if True, load the evaluators from the files and train
    them further. If False, train the evaluators from scratch.
    Trains a logistic evaluator for each stage of the game.
    The data is loaded using hardcoded filenames, and the models are saved
    to files with hardcoded names (all in the ./data directory).
    For each stage, we use the data not only from this stage, but also from
    the previous and from the next one. This allows us to have more data "for free".
    Returns a list with the 15 PatternLogisticEvaluator objects (and saves them).
    '''
    evaluators = [PatternLogisticEvaluator() for _ in range(15)]
    if load_evaluators:
        for i in range(15):
            evaluators[i].load(f'data/param_s{i}.npz')
        print("Loaded the evaluators from the files.")
    for i in range(15):
        X = np.concatenate(
            [np.load(f"data/X_s{j}.npy") for j in [i-1, i, i+1] if 0 <= j <= 14], 
            axis=0).astype(np.int32)
        y = np.concatenate(
            [np.load(f"data/y_s{j}.npy") for j in [i-1, i, i+1] if 0 <= j <= 14], 
            axis=0)
        print(f"Training stage {i} with {X.shape[0]} positions")
        evaluators[i].train(X, y, lr, iterations, batch_size)
        evaluators[i].save(f'data/param_s{i}.npz')
    return evaluators

def evaluate(position, maxval=10000, minval=-10000):
    '''
    position -- Position or BBPosition object
    maxval, minval: numbers representing the maximum and minimum possible values.
    Note : the default values are very large and essentially useless. Typically use 
    values around 64 or 100.

    Returns the score of the position as evaluated by the logistic evaluator.
    '''
    if isinstance(position, Position):
        empties = 64 - sum([1 for c in position.board if c != '.'])
        return max(minval, min(maxval, evaluators[empties // 4].evaluate(position)))
    else:
        empties = 64 - popcount(position.black | position.white)
        #evaluators is a global variable containing the 15 evaluators
        return max(minval, min(maxval, evaluators[empties // 4].evaluate_bb(position)))

#create_save_dataset(2001, 2023, stage=0) #approx 65k games, estimated runtime : 20min

'''
X = np.load('X_0_2001-2023.npy').astype(np.int32)
y = np.load('y_0_2001-2023.npy')
print(X.shape, X.dtype)
size = X.shape[0]
'''

'''
evaluator = PatternLogisticEvaluator()
#evaluator.train(X, y, learning_rate=0.001, iterations=size//128*2, batch_size=128)
#evaluator.save('data/weights_s0.npz')
evaluator.load('data/weights_s0.npz')
'''

# Check the learned values of some of the patterns : 
# For example, the first pattern, which looks at an edge and the two 
# X-squares next to it, should have a high value if it is filled with our disc, 
# and a low value if it is filled with the opponent's disc.
# If we take a pattern that's inbetween, for example the corners occupied by us
# and the rest by the opponent, the value should be somewhere in the middle.
'''
all_ours_idx = sum(3 ** i for i in range(10)) #1's everywhere
all_opp_idx = 2*all_ours_idx #2's everywhere
opp_wedged_idx = all_opp_idx - 3**0 - 3**7 #1's in the corners, 2's in the rest
us_wedged_idx = all_ours_idx + 3**0 + 3**7 #2's in the corners, 1's in the rest

print(f"Value of the pattern with all our discs : {evaluator.weights[0][all_ours_idx]}")
print(f"Value of the pattern with all opponent discs : {evaluator.weights[0][all_opp_idx]}")
print(f"Value for the opponent wedged : {evaluator.weights[0][opp_wedged_idx]}")
print(f"Value for us wedged : {evaluator.weights[0][us_wedged_idx]}")
'''

#create_dataset_from_edax_evals(2001, 2003, depth=10)
#train_all_stages(iterations=2000, lr=0.001, batch_size=256, load_evaluators=True)

evaluators = [PatternLogisticEvaluator() for _ in range(15)]
for s in range(15):
    evaluators[s].load(f'data/param_s{s}.npz')

