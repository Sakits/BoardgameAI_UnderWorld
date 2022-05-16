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

    g = Game(13, 5)

    n2 = NNet(g)
    n2.load_checkpoint('./checkpoint/', 'iteration-0014.pkl')
    args2 = dotdict({'num_MCTS_sims': 800, 'cpuct': 1.0, 'alpha':0.1, 'epsilon':0.25})
    mcts2 = MCTS(g, n2, args2)

    def n2p(x, turn):
        if turn <= 2:
            mcts2.reset()
        temp = 0
        # temp = 1 if turn <= 10 else 0
        policy = mcts2.getActionProb(x, temp=temp, turn=turn)
        step = np.random.choice(len(policy), p=policy)
        print(step // 15, step % 15)
        return step

    x = g.getInitBoard()
    x[7][7] = 1
    x = -np.array(x)
    # policy = mcts2.getActionProb(x, 0)
    # step = np.random.choice(len(policy), p=policy)
    # print(step // 15, step % 15)

    nnpolicy, value = n2.predict(g.getFeature(x))
    for j in range(15):
        for k in range(15):
            print('%.3f' % nnpolicy[j * 15 + k], end = " ")
        print(" ")
    g.display(x)
