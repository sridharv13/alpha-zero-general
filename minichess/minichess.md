
# Scaling Up Alpha Zero General (Minichess)


## Topics Covered

1. Initializing and displaying the Minichess Board
2. Checking whether game has ended or not
3. Defining players Player 1 and Player 2
4. Checking valid moves and mapping moves to Grid Points (to be enhanced to a3g5 format later)
5. Play all pieces one by one and validate (Pawn, Rook, King, Queen, Knight and Bishop)
6. Check Pawn cross attack
7. Check Pawn to reach last move and become Queuen
8. Complete the Game and check the winner



## Section 1 - Initializing and displaying the Minichess Board


- Load inital libraries


```python
from Coach import Coach
from minichess.GardnerMiniChessGame import GardnerMiniChessGame as Game
from minichess.MiniChessLogic import Board
from minichess.keras.NNet import NNetWrapper as nn
from utils import *
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
print(board)
```

    [[-479, -280, -320, -929, -60000], [-100, -100, -100, -100, -100], [0, 0, 0, 0, 0], [100, 100, 100, 100, 100], [479, 280, 320, 929, 60000]]


## Section 2 - Checking whether game has ended or not

- Verify the Canonical and User Board


```python
player1 = 1
player2 = -1
g.display(board)
```


      5 ♜ ♞ ♝ ♛ ♚
      4 ♟ ♟ ♟ ♟ ♟
      3 ·  ·  ·  ·  ·
      2 ♙ ♙ ♙ ♙ ♙
      1 ♖ ♘ ♗ ♕ ♔
        a  b  c  d  e




- Check whether Game is ended
- It should return 0 since Game is still in Valid state


```python
print(g.getGameEnded(board,player1))
```

    0


## Section 4 - Checking valid moves and mapping moves to Grid Points (to be enhanced to a3g5 format later)

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


    All possible moves are:

    [(36, 29), (37, 30), (38, 31), (39, 32), (40, 33), (44, 29), (44, 31)]

      5 ♜ ♞ ♝ ♛ ♚
      4 ♟ ♟ ♟ ♟ ♟
      3 ·  ·  ·  ·  ·
      2 ♙ ♙ ♙ ♙ ♙
      1 ♖ ♘ ♗ ♕ ♔
        a  b  c  d  e




- Execute first move by moving the White Pawn from a2 to a3

move = (36,29)


```python
logic.execute_move((36,29),player1)
g.display(logic.pieces_without_padding())
```


      5 ♖ ♘ ♗ ♕ ♔
      4 ·  ♙ ♙ ♙ ♙
      3 ♙ ·  ·  ·  ·
      2 ♟ ♟ ♟ ♟ ♟
      1 ♜ ♞ ♝ ♛ ♚
        a  b  c  d  e




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


    All possible moves are:

    [(36, 28), (37, 30), (37, 29), (38, 31), (39, 32), (40, 33), (40, 34), (43, 50), (43, 42), (44, 35), (44, 29), (44, 31), (44, 59), (44, 57), (44, 53), (44, 49), (45, 53), (45, 51), (46, 54), (46, 52), (46, 53), (47, 54), (47, 53), (47, 55), (47, 41), (47, 48)]

      5 ♖ ♘ ♗ ♕ ♔
      4 ·  ♙ ♙ ♙ ♙
      3 ♙ ·  ·  ·  ·
      2 ♟ ♟ ♟ ♟ ♟
      1 ♜ ♞ ♝ ♛ ♚
        a  b  c  d  e




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


      5 ♜ ·  ♝ ♛ ♚
      4 ♟ ♟ ♟ ♟ ♟
      3 ♞ ·  ·  ·  ·
      2 ·  ♙ ♙ ♙ ♙
      1 ♖ ♘ ♗ ♕ ♔
        a  b  c  d  e




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


    All possible moves are:

    [(37, 30), (37, 29), (38, 31), (39, 32), (40, 33), (43, 36), (43, 29), (44, 29), (44, 31)]

      5 ♜ ·  ♝ ♛ ♚
      4 ♟ ♟ ♟ ♟ ♟
      3 ♞ ·  ·  ·  ·
      2 ·  ♙ ♙ ♙ ♙
      1 ♖ ♘ ♗ ♕ ♔
        a  b  c  d  e





```python
logic.execute_move((44,29),player1)
g.display(logic.pieces_without_padding())
print('\nAll possible moves are: \n')
moves = []
for move in logic.get_legal_moves(player2):
    moves.append(move)
print(moves)
```


      5 ♖ ·  ♗ ♕ ♔
      4 ·  ♙ ♙ ♙ ♙
      3 ♘ ·  ·  ·  ·
      2 ♟ ♟ ♟ ♟ ♟
      1 ♜ ·  ♝ ♛ ♚
        a  b  c  d  e



    All possible moves are:

    [(36, 28), (37, 30), (37, 29), (38, 31), (39, 32), (40, 33), (40, 34), (43, 44), (43, 50), (43, 42), (45, 53), (45, 51), (46, 54), (46, 52), (46, 53), (47, 54), (47, 53), (47, 55), (47, 41), (47, 48)]


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


      5 ♜ ·  ♝ ♛ ♚
      4 ♟ ·  ♟ ♟ ♟
      3 ♟ ·  ·  ·  ·
      2 ·  ♙ ♙ ♙ ♙
      1 ♖ ·  ♗ ♕ ♔
        a  b  c  d  e



    All possible moves are:

    [(37, 30), (37, 23), (37, 29), (38, 31), (39, 32), (40, 33), (43, 36), (43, 29), (43, 44)]



```python
logic.execute_move((43,29),player1)
g.display(logic.pieces_without_padding())
print('\nAll possible moves are: \n')
moves = []
for move in logic.get_legal_moves(player2):
    moves.append(move)
print(moves)
```


      5 ·  ·  ♗ ♕ ♔
      4 ·  ♙ ♙ ♙ ♙
      3 ♖ ·  ·  ·  ·
      2 ♟ ·  ♟ ♟ ♟
      1 ♜ ·  ♝ ♛ ♚
        a  b  c  d  e



    All possible moves are:

    [(36, 28), (38, 31), (39, 32), (40, 33), (40, 34), (43, 44), (43, 50), (43, 42), (45, 53), (45, 51), (45, 37), (45, 29), (46, 54), (46, 52), (46, 53), (47, 54), (47, 53), (47, 55), (47, 41), (47, 48)]



```python
logic.execute_move((45,29),player2)
g.display(logic.pieces_without_padding())

print('\nAll possible moves for Player 1 are: \n')
moves = []
for move in logic.get_legal_moves(player1):
    moves.append(move)
print(moves)
```


      5 ♜ ·  ·  ♛ ♚
      4 ♟ ·  ♟ ♟ ♟
      3 ♝ ·  ·  ·  ·
      2 ·  ♙ ♙ ♙ ♙
      1 ·  ·  ♗ ♕ ♔
        a  b  c  d  e



    All possible moves for Player 1 are:

    [(37, 30), (37, 23), (37, 29), (38, 31), (39, 32), (40, 33)]



```python
logic.execute_move((37,29),player1)
g.display(logic.pieces_without_padding())

print('\nAll possible moves for Player 2 are: \n')
moves = []
for move in logic.get_legal_moves(player2):
    moves.append(move)
print(moves)
```


      5 ·  ·  ♗ ♕ ♔
      4 ·  ·  ♙ ♙ ♙
      3 ♙ ·  ·  ·  ·
      2 ♟ ·  ♟ ♟ ♟
      1 ♜ ·  ·  ♛ ♚
        a  b  c  d  e



    All possible moves for Player 2 are:

    [(36, 28), (38, 31), (39, 32), (40, 33), (40, 34), (43, 44), (43, 45), (43, 50), (43, 42), (46, 54), (46, 52), (46, 53), (46, 45), (46, 44), (47, 54), (47, 53), (47, 55), (47, 41), (47, 48)]



```python
logic.execute_move((46,44),player2)
g.display(logic.pieces_without_padding())

print('\nAll possible moves for Player 1 are: \n')
moves = []
for move in logic.get_legal_moves(player1):
    moves.append(move)
print(moves)
```


      5 ♜ ♛ ·  ·  ♚
      4 ♟ ·  ♟ ♟ ♟
      3 ♙ ·  ·  ·  ·
      2 ·  ·  ♙ ♙ ♙
      1 ·  ·  ♗ ♕ ♔
        a  b  c  d  e



    All possible moves for Player 1 are:

    [(38, 31), (39, 32), (40, 33), (45, 37)]



```python
logic.execute_move((39,32),player1)
g.display(logic.pieces_without_padding())

print('\nAll possible moves for Player 2 are: \n')
moves = []
for move in logic.get_legal_moves(player2):
    moves.append(move)
print(moves)
```


      5 ·  ·  ♗ ♕ ♔
      4 ·  ·  ♙ ·  ♙
      3 ♙ ·  ·  ♙ ·
      2 ♟ ·  ♟ ♟ ♟
      1 ♜ ♛ ·  ·  ♚
        a  b  c  d  e



    All possible moves for Player 2 are:

    [(36, 28), (38, 31), (38, 32), (40, 33), (40, 32), (40, 34), (43, 50), (43, 42), (44, 52), (44, 50), (44, 37), (44, 30), (44, 23), (44, 16), (44, 9), (44, 45), (44, 46), (44, 51), (47, 54), (47, 53), (47, 55), (47, 41), (47, 46), (47, 48)]



```python
logic.execute_move((44,37),player2)
g.display(logic.pieces_without_padding())

print('\nAll possible moves for Player 1 are: \n')
moves = []
for move in logic.get_legal_moves(player1):
    moves.append(move)
print(moves)
```


      5 ♜ ·  ·  ·  ♚
      4 ♟ ♛ ♟ ♟ ♟
      3 ♙ ·  ·  ♙ ·
      2 ·  ·  ♙ ·  ♙
      1 ·  ·  ♗ ♕ ♔
        a  b  c  d  e



    All possible moves for Player 1 are:

    [(29, 23), (32, 24), (32, 26), (38, 31), (40, 33), (45, 37), (45, 39), (45, 33), (46, 39), (47, 39)]


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


      5 ·  ·  ♗ ♕ ♔
      4 ·  ·  ♙ ·  ♙
      3 ♙ ·  ·  ·  ·
      2 ♟ ♛ ♙ ♟ ♟
      1 ♜ ·  ·  ·  ♚
        a  b  c  d  e



    All possible moves for Player 2 are:

    [(36, 28), (37, 45), (37, 53), (37, 29), (37, 31), (37, 25), (37, 19), (37, 30), (37, 23), (37, 16), (37, 9), (37, 38), (37, 44), (37, 51), (39, 32), (39, 25), (40, 33), (40, 34), (43, 44), (43, 45), (43, 46), (47, 45), (43, 50), (43, 42), (47, 54), (47, 53), (47, 55), (47, 41), (47, 46), (47, 48)]



```python
logic.execute_move((37,25),player2)
g.display(logic.pieces_without_padding())

print('\nAll possible moves for Player 1 are: \n')
moves = []
for move in logic.get_legal_moves(player1):
    moves.append(move)
print(moves)
```


      5 ♜ ·  ·  ·  ♚
      4 ♟ ·  ♙ ♟ ♟
      3 ♙ ·  ·  ·  ·
      2 ·  ·  ♙ ♛ ♙
      1 ·  ·  ♗ ♕ ♔
        a  b  c  d  e



    All possible moves for Player 1 are:

    [(24, 17), (38, 31), (40, 33), (45, 37), (45, 39), (46, 39), (47, 39)]



```python
logic.execute_move((24,17),player1)
g.display(logic.pieces_without_padding())

print('\nAll possible moves for Player 2 are: \n')
moves = []
for move in logic.get_legal_moves(player2):
    moves.append(move)
print(moves)
```


      5 ·  ·  ♗ ♕ ♔
      4 ·  ·  ♙ ♛ ♙
      3 ♙ ·  ·  ·  ·
      2 ♟ ·  ·  ♟ ♟
      1 ♜ ·  ♕ ·  ♚
        a  b  c  d  e



    All possible moves for Player 2 are:

    [(25, 33), (25, 41), (25, 31), (25, 37), (25, 17), (25, 19), (25, 18), (25, 26), (25, 32), (25, 24), (36, 28), (39, 32), (40, 33), (40, 34), (43, 44), (43, 45), (43, 50), (43, 42), (47, 54), (47, 53), (47, 55), (47, 41), (47, 46), (47, 48)]


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


      5 ♜ ·  ♕ ·  ♚
      4 ♟ ·  ·  ♟ ♟
      3 ♙ ·  ·  ·  ·
      2 ·  ·  ♙ ·  ♙
      1 ·  ·  ♗ ♕ ♛
        a  b  c  d  e



    All possible moves for Player 1 are:

    [(17, 25), (17, 23), (17, 18), (17, 19), (17, 24), (17, 31), (17, 16), (17, 15), (38, 31), (38, 24), (40, 33), (45, 37), (45, 39), (45, 33), (46, 39), (46, 32), (46, 25), (46, 47)]



```python
print(logic.is_win(player2))
print(logic.is_win(player1))
print(g.getGameEnded(logic.pieces_without_padding(),player2))
```

    True
    False
    0

