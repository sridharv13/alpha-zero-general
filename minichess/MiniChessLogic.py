"""
Board class for the game of MiniChess.
Default board size is 5x5.

Author: Karthik selvakumar, github.com/karthikselva
Date: May 15, 2018.

"""

from __future__ import print_function
import re, sys, time
from itertools import count
from collections import OrderedDict, namedtuple
import numpy as np
from enum import Enum


class Board:

    PAWN = 100
    KNIGHT = 280
    BISHOP = 320
    ROOK = 479
    QUEEN = 929
    KING = 60000
    BLANK = 0

    def __init__(self, n, pieces):
        "Set up initial board configuration."
        self.n = n
        self.last_cell = n*n - 1
        self.bottom_left = n*n - n
        self.bottom_right = self.last_cell
        self.top_left = 0
        self.top_right = n - 1
        self.pieces = pieces
        self.player_won = 0
        self.wc = (True, True)
        self.bc = (True, True)
        self.ep = 0
        self.kp = 0

        self.north, self.east, self.south, self.west = -(self.n), 0, (self.n), -1
        self.directions = {
            Board.PAWN: (self.north, self.north + self.north, self.north + self.west, self.north + self.east),
            Board.KNIGHT: (self.north + self.north + self.east, self.east + self.north + self.east,
                  self.east + self.south + self.east, self.south + self.south + self.east,
                  self.south + self.south + self.west, self.west + self.south + self.west,
                  self.west + self.north + self.west, self.north + self.north + self.west),
            Board.BISHOP: (self.north + self.east, self.south + self.east,
                  self.south + self.west, self.north + self.west),
            Board.ROOK: (self.north, self.east, self.south, self.west),
            Board.QUEEN: (self.north, self.east, self.south, self.west, self.north + self.east,
                  self.south + self.east, self.south + self.west, self.north + self.west),
            Board.KING: (self.north, self.east, self.south, self.west, self.north + self.east,
                  self.south + self.east, self.south + self.west, self.north + self.west)
        }

    def get_legal_moves(self,player):
        # For each of our pieces, iterate through each possible 'ray' of moves,
        # as defined in the 'directions' map. The rays are broken e.g. by
        # captures or immediately in case of pieces such as knights.
        flat_pieces = [item for sublist in self.pieces for item in sublist]
        for i, p in enumerate(flat_pieces):
            if p <= 0: continue
            print(self.directions[p])
            for d in self.directions[p]:
                for j in count(i+d, d):
                    if j >= self.n: break
                    q = flat_pieces[j]
                    # Stay inside the board, and off friendly pieces
                    if q == Board.BLANK or q > 0: break
                    # Pawn move, double move and capture
                    if p == Board.PAWN and d in (self.north, self.north+self.north) and q != Board.BLANK: break
                    if p == Board.PAWN and d == self.north+self.north and (i < self.bottom_left+self.north or flat_pieces[i+self.north] != Board.BLANK): break
                    if p == Board.PAWN and d in (self.north+self.west, self.north+self.east) and q == Board.BLANK and j not in (self.ep, self.kp): break
                    # Move it
                    yield (i, j)
                    # Stop crawlers from sliding, and sliding after captures
                    if p in [Board.PAWN,Board.KNIGHT,Board.KING] or q < 0: break
                    # Castling, by sliding the rook next to the king
                    if i == self.bottom_left and flat_pieces[j+self.east] == Board.KING and self.wc[0]: yield (j+self.east, j+self.west)
                    if i == self.bottom_right and flat_pieces[j+self.west] == Board.KING and self.wc[1]: yield (j+self.west, j+self.east)

    def rotate(self):
        ''' Rotates the board, preserving enpassant '''
        return (
            self.pieces*(-1),self.bc, self.wc,
            self.last_cell-self.ep if self.ep else 0,
            self.last_cell-self.kp if self.kp else 0)


    def execute_move(self, move, player):
        i, j = move
        i = i
        flat_pieces = [item for sublist in self.pieces for item in sublist]
        p, q = flat_pieces[i], flat_pieces[j]
        put = lambda board, i, p: np.append(np.append(board[:i],[p]),board[i+1:])
        # Copy variables and reset ep and kp
        board = self.pieces
        wc, bc, ep, kp = self.wc, self.bc, 0, 0
        # Actual move
        if abs(flat_pieces[j]) == Board.KING:
            self.player_won = player
        board = put(board, j, board[i])
        board = put(board, i, Board.BLANK)
        # Castling rights, we move the rook or capture the opponent's
        if i == self.bottom_left: wc = (False, wc[1])
        if i == self.bottom_right: wc = (wc[0], False)
        if j == self.top_left: bc = (bc[0], False)
        if j == self.top_right: bc = (False, bc[1])
        # Castling
        if p == Board.KING:
            wc = (False, False)
            if abs(j-i) == 2:
                kp = (i+j)//2
                board = put(board, self.bottom_left if j < i else self.bottom_right, '.')
                board = put(board, kp, Board.ROOK)
        # Pawn promotion, double move and en passant capture
        if p == Board.PAWN:
            if self.top_left <= j <= self.top_right:
                board = put(board, j, Board.QUEEN)
            if j - i == 2*self.north:
                ep = i + self.north
            if j - i in (self.north+self.west, self.north+self.east) and q == Board.BLANK:
                board = put(board, j+self.south, Board.BLANK)
        # We rotate the returned position, so it's ready for the next player
        return self.rotate()


    def has_legal_moves(self,player):
        for move in self.get_legal_moves(player):
            return True
        return False

    def is_win(self, player):
        if player > 0 and self.player_won == player:
            return True
        if player < 0 and self.player_won == player:
            return True
        return False

    # add [][] indexer syntax to the Board
    def __getitem__(self, index):
        return self.pieces[index]

    def display(self):
        print()
        uni_pieces = {
            -Board.ROOK: '♜',
            -Board.KNIGHT: '♞',
            -Board.BISHOP: '♝',
            -Board.QUEEN: '♛',
            -Board.KING: '♚',
            -Board.PAWN: '♟',
            Board.ROOK: '♖',
            Board.KNIGHT: '♘',
            Board.BISHOP: '♗',
            Board.QUEEN: '♕',
            Board.KING: '♔',
            Board.PAWN: '♙',
            Board.BLANK: '· '
        }
        for i, row in enumerate(self.pieces.split()):
            print(' ', self.n - i, ' '.join(uni_pieces.get(p, p) for p in row))
        print('    a  b  c  d  e  \n\n')