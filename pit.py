import numpy as np
from utils import *

from train.Arena import Arena
from game.Players import *
from mcts.MCTSWrapper import MCTSWrapper as MCTS
from network.NNetWrapper import NNetWrapper as NNet
from libcpp import Game

"""
use this script to play any two agents against each other, or play manually with
any agent.
"""
if __name__ == '__main__':

    g = Game(9, 5)

    # all players
    # rp = RandomPlayer(g).play
    # np = NNPlayer(g).play
    # hp = HumanPlayer(g).play

    # nnet players
    n1 = NNet(g)
    n1.load_checkpoint('./checkpoint/', 'iteration-0000.pkl')
    args1 = dotdict({'num_MCTS_sims': 500, 'cpuct': 1.0})
    mcts1 = MCTS(g, n1, args1)

    def n1p(x, turn):
        if turn <= 2:
            mcts1.reset()
        temp = 0
        # temp = 1 if turn <= 10 else 0
        policy = mcts1.getActionProb(x, temp=temp)
        return np.random.choice(len(policy), p=policy)

    n2 = NNet(g)
    n2.load_checkpoint('./checkpoint/', 'iteration-0000.pkl')
    args2 = dotdict({'num_MCTS_sims': 500, 'cpuct': 1.0})
    mcts2 = MCTS(g, n2, args2)

    def n2p(x, turn):
        if turn <= 2:
            mcts2.reset()
        temp = 0
        # temp = 1 if turn <= 10 else 0
        policy = mcts2.getActionProb(x, temp=temp)
        return np.random.choice(len(policy), p=policy)

    arena = Arena(n1p, n2p, g, display=g.display)
    print(arena.playGames(2, verbose=True))
