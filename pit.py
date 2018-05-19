import Arena
from MCTS import MCTS
from minichess.GardnerMiniChessGame import GardnerMiniChessGame, display
from minichess.MiniChessPlayer import *
from minichess.keras.NNet import NNetWrapper as NNet

import numpy as np
from utils import *

"""
use this script to play any two agents against each other, or play manually with
any agent.
"""

g = GardnerMiniChessGame(5)

# all players
rp1 = RandomPlayer(g).play
rp2 = RandomPlayer(g).play

# nnet players
n1 = NNet(g)
n1.load_checkpoint('./pretrained_models/minichess/keras/','temp.pth.tar')
args1 = dotdict({'numMCTSSims': 50, 'cpuct':1.0})
mcts1 = MCTS(g, n1, args1)
n1p = lambda x: np.argmax(mcts1.getActionProb(x, temp=0))

arena = Arena.Arena(n1p, rp2, g, display=display)
print(arena.playGames(20, verbose=False))
