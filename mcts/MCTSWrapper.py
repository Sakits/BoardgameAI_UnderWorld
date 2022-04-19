from libcpp import MCTS
import numpy as np

class MCTSWrapper():
    
    def __init__(self, game, nnet, args):
        self.action = game.getActionSize()
        self.alpha = args.alpha
        self.mcts = MCTS(game, nnet.predict if nnet else None, args.num_MCTS_sims, args.cpuct, args.epsilon)
        self.reset()

    def reset(self):
        self.mcts.reset()

    def getActionProb(self, canonicalBoard, temp = 1):
        return self.mcts.getActionProb(canonicalBoard, temp)

    def getExpertProb(self, canonicalBoard, temp=1, prune=False):
        return self.mcts.getExpertProb(canonicalBoard, temp, prune)

    def processResults(self, pi, value):
        self.mcts.processResult(pi, value)

    def findLeafToProcess(self, canonicalBoard, isRoot):
        flag, ans = self.mcts.findLeafToProcess(canonicalBoard, np.random.dirichlet([self.alpha] * self.action))
        return ans if flag else None
