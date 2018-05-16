from __future__ import print_function
import sys
sys.path.append('..')
from Game import Game
from .MiniChessLogic import Board
import numpy as np

"""
Game class implementation for the game of TicTacToe.

Author: Karthik selvakumar, github.com/karthikselva
Date: May 15, 2018.
"""

class GardnerMiniChessGame(Game):
    def __init__(self, n=5):
        self.n = n

    def getInitBoard(self):
        # return initial board (numpy board)
        b = Board(self.n,
            [
                [Board.ROOK,  Board.KNIGHT,  Board.BISHOP,  Board.QUEEN, Board.KING],
                [Board.PAWN,  Board.PAWN,    Board.PAWN,    Board.PAWN,  Board.PAWN],
                [Board.BLANK, Board.BLANK,   Board.BLANK,   Board.BLANK, Board.BLANK],
                [-Board.PAWN, -Board.PAWN,   -Board.PAWN,   -Board.PAWN, -Board.PAWN],
                [-Board.ROOK, -Board.KNIGHT, -Board.BISHOP, -Board.QUEEN,-Board.KING],
            ]
        )
        return np.array(b.pieces)

    def getBoardSize(self):
        # (a,b) tuple
        return (self.n, self.n)

    def getActionSize(self):
        # return number of actions
        return self.n*self.n + 1

    def getNextState(self, board, player, action):
        # if player takes action on board, return next (board,player)
        # action must be a valid move
        if action == self.n*self.n:
            return (board, -player)
        b = Board(self.n,board)
        move = (int(action/self.n), action%self.n)
        b.execute_move(move,player)
        return (b.pieces, -player)

    def getValidMoves(self, board, player):
        # return a fixed size binary vector
        valids = [0]*self.getActionSize()
        b = Board(self.n,board)
        legalMoves = b.get_legal_moves(player)
        if not b.has_legal_moves(player):
            valids[-1]=1
            return np.array(valids)
        for x, y in legalMoves:
            try:
                valids[self.n*x+y]=1
            except:
                print("x: " + str(x) + " y: " + str(y))
        return np.array(valids)

    def getGameEnded(self, board, player):
        # return 0 if not ended, 1 if player 1 won, -1 if player 1 lost
        # player = 1
        b = Board(self.n,board)

        if b.is_win(player):
            return 1
        if b.is_win(-player):
            return -1
        if b.has_legal_moves(player):
            return 0
        # draw has a very little value
        return 1e-4

    def getCanonicalForm(self, board, player):
        # return state if player==1, else return -state if player==-1
        return player*board

    def getSymmetries(self, board, pi):
        # mirror, rotational
        assert(len(pi) == self.n**2+1)  # 1 for pass
        pi_board = np.reshape(pi[:-1], (self.n, self.n))
        l = []

        for i in range(1, 5):
            for j in [True, False]:
                newB = np.rot90(board, i)
                newPi = np.rot90(pi_board, i)
                if j:
                    newB = np.fliplr(newB)
                    newPi = np.fliplr(newPi)
                l += [(newB, list(newPi.ravel()) + [pi[-1]])]
        return l

    def stringRepresentation(self, board):
        return np.array_str(np.array(board))

    def display(board):
        board.display()
