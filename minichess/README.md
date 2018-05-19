# Scaling Up Alpha Zero General (Minichess)

An implementation of a simple game provided to check extendability of the framework. Main difference of this game comparing to Othello is that it allows draws, i.e. the cases when nobody won after the game ended. To support such outcomes ```Arena.py``` and ```Coach.py``` classes were modified. Neural network architecture was copy-pasted from the game of Othello, so possibly it can be simplified.

To train a model for MiniChess, change the imports in ```main.py``` to:
```python
from Coach import Coach
from minichess.MiniChessGame import Game
from minichess.keras.NNet import NNetWrapper as nn
from utils import *
```

 Make similar changes to ```pit.py```.

To start training a model for TicTacToe:
```bash
python main.py
```
To start a tournament of 10 episodes with the model-based player against a random player:
```bash
python pit.py
```
You can check the results of RandomPlayer vs Neural Network Player in ```pit.py```

### Experiments
I trained a Keras model for 5X5 Minichess (10 iterations, 25 episodes, 10 epochs per iteration and 25 MCTS simulations per turn). This took about 15 minutes on an i7-7330 with CUDA Nvida GTX 960. The pretrained model (Keras) can be found in ```pretrained_models/minichess/keras/```.

## Implementation Details

1. Initializing and displaying the Minichess Board
2. Checking whether game has ended or not
3. Defining players Player 1 and Player 2
4. Checking valid moves and mapping moves to Grid Points (to be enhanced to a3g5 format later)
5. Play all pieces one by one and validate (Pawn, Rook, King, Queen, Knight and Bishop)
6. Check Pawn cross attack
7. Check Pawn to reach last move and become Queuen
8. Complete the Game and check the winner
9. Normalize Actions to Ids
10. Valid moves in MCTS
11. Train with Neural Network



## Section 1 - Initializing and displaying the Minichess Board


- Load inital libraries


```python
from Coach import Coach
from minichess.GardnerMiniChessGame import GardnerMiniChessGame as Game
from minichess.MiniChessLogic import Board
from minichess.keras.NNet import NNetWrapper as nn
from utils import *
import numpy as np
```

    c:\users\karthik\appdata\local\programs\python\python36\lib\site-packages\h5py\__init__.py:36: FutureWarning: Conversion of the second argument of issubdtype from `float` to `np.floating` is deprecated. In future, it will be treated as `np.float64 == np.dtype(float).type`.
      from ._conv import register_converters as _register_converters
    Using TensorFlow backend.


- Load the Game and required Neural Network


```python
g = Game()
nnet = nn(g)
board = g.getInitBoard()
n = 5 # 5X5 Grid
logic = Board(n,board)
print(np.array(board))
```

    [[  -479   -280   -320   -929 -60000]
     [  -100   -100   -100   -100   -100]
     [     0      0      0      0      0]
     [   100    100    100    100    100]
     [   479    280    320    929  60000]]


## Section 2 - Checking whether game has ended or not

- Verify the Canonical and User Board


```python
player1 = 1
player2 = -1
g.display(board)
```


    ---------------------------------------------------------------------------

    TypeError                                 Traceback (most recent call last)

    <ipython-input-3-1c914fc1972b> in <module>()
          1 player1 = 1
          2 player2 = -1
    ----> 3 g.display(board)


    TypeError: display() missing 1 required positional argument: 'player'


- Check whether Game is ended
- It should return 0 since Game is still in Valid state


```python
print(g.getGameEnded(board,player1))
```

## Section 3 - Checking valid moves and mapping moves to Grid Points (to be enhanced to a3g5 format later)

- Check the next legal move we can do
- It Should list all Pawn moves for White and Horse moves

[(36, 29), (37, 30), (38, 31), (39, 32), (40, 33), (44, 29), (44, 31)]



Overal Chess Grid 5X5 Looks like this:

        # Chess GRID with Padding and Cell Number
        # [0,  1,  2,  3,  4,  5,  6]
        # [7,  8,  9,  10, 11, 12, 13]

        # [14,   15, 16, 17, 18, 19,     20]
        # [21,   22, 23, 24, 25, 26,     27]
        # [28,   29, 30, 31, 32, 33,     34]
        # [35,   36, 37, 38, 39, 40,     41]
        # [42,   43, 44, 45, 46, 47,     48]

        # [49, 50, 51, 52, 53, 54, 55]
        # [56, 57, 58, 59, 60, 61, 62]



```python
print('\nAll possible moves are: \n')
moves = []
for move in logic.get_legal_moves(player1):
    moves.append(move)
print(moves)
g.display(board)
```

- Execute first move by moving the White Pawn from a2 to a3

move = (36,29)


```python
logic.execute_move((36,29),player1)
g.display(logic.pieces_without_padding())
```

- Assign Player 2 as -1
- Get Legal moves for Player 2
- Execute one pawn Move


```python
player2 = -1
print('\nAll possible moves are: \n')
moves = []
for move in logic.get_legal_moves(player2):
    moves.append(move)
print(moves)
g.display(logic.pieces_without_padding())
```

## Section 4 - Checking valid moves and mapping moves to Grid Points (to be enhanced to a3g5 format later)

Overal Chess Grid 5X5 Looks like this:

        # Chess GRID with Padding and Cell Number
        # [0,  1,  2,  3,  4,  5,  6]
        # [7,  8,  9,  10, 11, 12, 13]

        # [14,   15, 16, 17, 18, 19,     20]
        # [21,   22, 23, 24, 25, 26,     27]
        # [28,   29, 30, 31, 32, 33,     34]
        # [35,   36, 37, 38, 39, 40,     41]
        # [42,   43, 44, 45, 46, 47,     48]

        # [49, 50, 51, 52, 53, 54, 55]
        # [56, 57, 58, 59, 60, 61, 62]

Now make a horse move from b1 to a3


```python
logic.execute_move((44,29),player2)
g.display(logic.pieces_without_padding())
```

- White to take the horse
- List legal moves and take the horse


```python
print('\nAll possible moves are: \n')
moves = []
for move in logic.get_legal_moves(player1):
    moves.append(move)
print(moves)
g.display(logic.pieces_without_padding())
```


```python
logic.execute_move((44,29),player1)
g.display(logic.pieces_without_padding())
print('\nAll possible moves are: \n')
moves = []
for move in logic.get_legal_moves(player2):
    moves.append(move)
print(moves)
```

## Section 5 - Play all pieces one by one and validate (Pawn, Rook, King, Queen, Knight and Bishop)

Refer the Grid to Make a move


Overal Chess Grid 5X5 Looks like this:

        # Chess GRID with Padding and Cell Number
        # [0,  1,  2,  3,  4,  5,  6]
        # [7,  8,  9,  10, 11, 12, 13]

        # [14,   15, 16, 17, 18, 19,     20]
        # [21,   22, 23, 24, 25, 26,     27]
        # [28,   29, 30, 31, 32, 33,     34]
        # [35,   36, 37, 38, 39, 40,     41]
        # [42,   43, 44, 45, 46, 47,     48]

        # [49, 50, 51, 52, 53, 54, 55]
        # [56, 57, 58, 59, 60, 61, 62]




```python
logic.execute_move((37,29),player2)
g.display(logic.pieces_without_padding())
print('\nAll possible moves are: \n')
moves = []
for move in logic.get_legal_moves(player1):
    moves.append(move)
print(moves)
```


```python
logic.execute_move((43,29),player1)
g.display(logic.pieces_without_padding())
print('\nAll possible moves are: \n')
moves = []
for move in logic.get_legal_moves(player2):
    moves.append(move)
print(moves)
```


```python
logic.execute_move((45,29),player2)
g.display(logic.pieces_without_padding())

print('\nAll possible moves for Player 1 are: \n')
moves = []
for move in logic.get_legal_moves(player1):
    moves.append(move)
print(moves)
```


```python
logic.execute_move((37,29),player1)
g.display(logic.pieces_without_padding())

print('\nAll possible moves for Player 2 are: \n')
moves = []
for move in logic.get_legal_moves(player2):
    moves.append(move)
print(moves)
```


```python
logic.execute_move((46,44),player2)
g.display(logic.pieces_without_padding())

print('\nAll possible moves for Player 1 are: \n')
moves = []
for move in logic.get_legal_moves(player1):
    moves.append(move)
print(moves)
```


```python
logic.execute_move((39,32),player1)
g.display(logic.pieces_without_padding())

print('\nAll possible moves for Player 2 are: \n')
moves = []
for move in logic.get_legal_moves(player2):
    moves.append(move)
print(moves)
```


```python
logic.execute_move((44,37),player2)
g.display(logic.pieces_without_padding())

print('\nAll possible moves for Player 1 are: \n')
moves = []
for move in logic.get_legal_moves(player1):
    moves.append(move)
print(moves)
```

## Section 7 -  Check Pawn to reach last move and become Queuen



```python
logic.execute_move((32,24),player1)
g.display(logic.pieces_without_padding())

print('\nAll possible moves for Player 2 are: \n')
moves = []
for move in logic.get_legal_moves(player2):
    moves.append(move)
print(moves)
```


```python
logic.execute_move((37,25),player2)
g.display(logic.pieces_without_padding())

print('\nAll possible moves for Player 1 are: \n')
moves = []
for move in logic.get_legal_moves(player1):
    moves.append(move)
print(moves)
```


```python
logic.execute_move((24,17),player1)
g.display(logic.pieces_without_padding())

print('\nAll possible moves for Player 2 are: \n')
moves = []
for move in logic.get_legal_moves(player2):
    moves.append(move)
print(moves)
```

## Section 8 - Complete the Game and check the winner



```python
logic.execute_move((25,19),player2)
g.display(logic.pieces_without_padding())

print('\nAll possible moves for Player 1 are: \n')
moves = []
for move in logic.get_legal_moves(player1):
    moves.append(move)
print(moves)

```


```python
print(logic.is_win(player2))
print(logic.is_win(player1))
print(g.getGameEnded(logic.pieces_without_padding(),player2))
```

## Section 9 - Normalize Actions to Ids

- Every piece can take any position in 5 X 5 Grid
- Queen Can move diagonally and straig from every cell (0,0), (0,1) .... (3,1) ... (4,4)
- Similarly every piece can move to all possible positions constrained by their rule
- We will store two hash maps
- Action Identifier to Actions   { 51: ["Queen", "Cell 1", "Cell 10"], ... }
- Action to Action Identifier    { "Queen:Cell1:Cell10": 242, ... }


```python
print(g.id_to_action[100])
print(g.action_to_id["479:26:12"])
```

- Action size really huge compared to other board games because of different piece type and moves
- This grows exponentially high once the board size starts growing (say n = 8)


```python
print(len(g.id_to_action))
```

## Section 10 - Get All Valid Moves in the MCTS consumable format

Overal Chess Grid 5X5 Looks like this:

        # Chess GRID with Padding and Cell Number
        # [0,  1,  2,  3,  4,  5,  6]
        # [7,  8,  9,  10, 11, 12, 13]

        # [14,   15, 16, 17, 18, 19,     20]
        # [21,   22, 23, 24, 25, 26,     27]
        # [28,   29, 30, 31, 32, 33,     34]
        # [35,   36, 37, 38, 39, 40,     41]
        # [42,   43, 44, 45, 46, 47,     48]

        # [49, 50, 51, 52, 53, 54, 55]
        # [56, 57, 58, 59, 60, 61, 62]



```python
import numpy as np
g = Game()
nnet = nn(g)
board = g.getInitBoard()
n = 5 # 5X5 Grid
logic = Board(n,board)
print(np.array(board))
print('\nAll possible moves for Player 2 are: \n')
moves = []
for move in logic.get_legal_moves(player2):
    moves.append(move)
print(moves)
print(len(g.action_to_id))
valids = g.getValidMoves(board,player1)
print(len(valids))
count = 0
for i in valids:
    if i == 1: count += 1
print(count)
print(len(moves))

```

## Section 11 - Train with Neural Network

- Changed Neural Network activation from softmax to sigmoid due to vanishing gradients
- Never use numpy array and python array mixed
- Rotation and player switching in canonical board might be conufusing


```python
from Coach import Coach
from minichess.GardnerMiniChessGame import GardnerMiniChessGame as Game
from minichess.keras.NNet import NNetWrapper as nn
from utils import *

args = dotdict({
    'numIters': 10,
    'numEps': 5,
    'tempThreshold': 15,
    'updateThreshold': 0.6,
    'maxlenOfQueue': 200,
    'arenaCompare': 10,
    'numMCTSSims': 25,
    'cpuct': 1,
    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('/dev/models/8x100x50', 'best.pth.tar'),
    'numItersForTrainExamplesHistory': 20,
})

g = Game()
nnet = nn(g)
c = Coach(g, nnet, args)

```


```python
c.executeEpisode()
print()
```


    ♜ ♞ ♝ ♛ ♚
    ♟ ♟ ♟ ♟ ♟
    ·  ·  ·  ·  ·
    ♙ ♙ ♙ ♙ ♙
    ♖ ♘ ♗ ♕ ♔

    ♜ ♞ ♝ ♛ ♚
    ♟ ♟ ♟ ♟ ♟
    ♙ ·  ·  ·  ·
    ·  ♙ ♙ ♙ ♙
    ♖ ♘ ♗ ♕ ♔

    ♜ ·  ♝ ♛ ♚
    ♟ ♟ ♟ ♟ ♟
    ♙ ·  ♞ ·  ·
    ·  ♙ ♙ ♙ ♙
    ♖ ♘ ♗ ♕ ♔

    ♜ ·  ♝ ♛ ♚
    ♟ ♟ ♟ ♟ ♟
    ♙ ♙ ♞ ·  ·
    ·  ·  ♙ ♙ ♙
    ♖ ♘ ♗ ♕ ♔

    ♜ ·  ♝ ♛ ♚
    ♟ ♟ ♟ ♟ ♟
    ♙ ♙ ·  ·  ·
    ·  ·  ♙ ♙ ♞
    ♖ ♘ ♗ ♕ ♔


### Contributors and Credits
* [Karthik selvakumar Bhuvaneswaran](https://github.com/karthikselva)

The implementation is based on the game of Othello (https://github.com/suragnair/alpha-zero-general/tree/master/othello).

### AlphaGo / AlphaZero Events
* February 8, 2018 - [Solving Alpha Go Zero + TensorFlow, Kubernetes-based Serverless AI Models on GPU](https://www.meetup.com/Advanced-Spark-and-TensorFlow-Meetup/events/245308722/)
