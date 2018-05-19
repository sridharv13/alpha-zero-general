import numpy as np

"""
Random player for the game of MiniChess.

Author: Karthik selvakumar, github.com/karthikselva
Date: May 15, 2018.

"""
class RandomPlayer():
    def __init__(self, game):
        self.game = game

    def play(self, board):
        a = np.random.randint(self.game.getActionSize())
        valids = self.game.getValidMoves(board, 1)
        while valids[a]!=1:
            a = np.random.randint(self.game.getActionSize())
        return a