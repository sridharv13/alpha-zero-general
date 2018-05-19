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
    'numMCTSSims': 3,
    'cpuct': 1,
    'checkpoint': './temp/',
    'load_model': False,
    'load_folder_file': ('/dev/models/8x100x50', 'best.pth.tar'),
    'numItersForTrainExamplesHistory': 20,
})

if __name__ == "__main__" :
    g = Game()
    nnet = nn(g)

    if args.load_model:
        nnet.load_checkpoint(args.load_folder_file[0], args.load_folder_file[1])

    c = Coach(g, nnet, args)
    if args.load_model:
        print("Load trainExamples from file")
        c.loadTrainExamples()
    c.learn()
