## Introduction

TinyOthello is a small python-based Othello engine. It was inspired by [sunfish](https://github.com/thomasahle/sunfish), a chess engine written hardly over 100 lines of code. I started by implementing the same functionalities, and then slowly made it more and more complex by improving the different components of the engine.  

The goal of this project is for me to write my first Othello engine from start to finish, and to experiment with the SOTA (or not so SOTA) algorithms in the area. I plan on experimenting with more search algorithms and maybe some of my own ideas later.

## Usage

You can run the engine with the command `python tinyothello.py`. The short list of available commands can be printed with the command `help`. For example, you can play the first move using `move f5`. You can also play against the engine (with either color), ask the engine to evaluate a position, change the depth of evaluations, etc.  

Otherwise, it is also possible to use the engine with the [nboard](https://github.com/weltyc/nboard) GUI, as an analysis engine. Simply add an analysis engine, and select the folder `tinyothello/`, with the run command `python tinyothello-nboard.py`.

## Features

### Move generation

At first the move generation was completely naive (and is located in the `game.py` file). However, it was later upgraded to a bitboard-based implementation for performance purposes. The move generation uses pre-computed tables for each row/column/diagonal, but figuring out the discs to flip for a given move is still computed entirely on the move. 

### Search

The search algorithm is the same as the one in [sunfish](https://github.com/thomasahle/sunfish), i.e. [NegaC*](https://www.chessprogramming.org/index.php?title=NegaC*&mobileaction=toggle_view_desktop). It basically uses null-window alphabeta searches to perform a binary search on the possible evaluation values of the position : say that the evaluations must be between -64 and 64. You would start by calculating whether it is above or below 0. If it is above, you will then calculate whether it is above or below 32 (the new middle point). After enough iterations, you figure out the true evaluation.

A basic transposition table is implemented, although it uses basic python hashing instead of Zobrist hashing. This will be improved at some point.

### Evaluation function

The search algorithm needs an evaluation function ; a function that can map any position to an evaluation. TinyOthello uses a [pattern-based evaluation function](https://skatgame.net/mburo/ps/evalfunc.pdf), trained on positions of the [WTHOR](https://www.ffothello.org/informatique/la-base-wthor/) database that I evaluated using the [edax](https://github.com/abulmo/edax-reversi) engine. I could have trained my evaluation function without relying on another program by first calculating a dataset of perfectly solved endgame positions, then training a very strong endgame evaluation function using this dataset, and then using this new evaluation function to create a new dataset of midgame positions, etc. However, this process would have taken too much time for such a "simple" project, so I decided to go the easy route and use an already existing engine to create my dataset of evaluated positions.

The evaluation function is currently rather weak due to a lack of computing power : I haven't let the code run for a long time yet, so the dataset used to train the evaluation function is still very small, which is suboptimal. 

### How strong is it ?

It is not insanely strong yet due to the very small dataset used to train the evaluation function. It already does a decent job at choosing the actual good moves and giving bad evaluations to the actual bad moves in most positions, however the lack of data and training makes the value of the evaluations essentially meaningless...


### Python = slow ?

Python is definitely not the best language for such an engine due to the nature of the algorithms (computation-heavy, lots of function calls, loops, etc.), but I wanted something easy to write to start with. I did use the [Numba](https://numba.pydata.org/) library to JIT some of the "primitive" functions used for move generation though.


