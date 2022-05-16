from libcpp import MCTS
import numpy as np

class MCTSWrapper():
    
    def __init__(self, game, nnet, args):
        self.action = game.getActionSize()
        self.mcts = MCTS(game, nnet.predict if nnet else None, args.num_MCTS_sims, args.cpuct, args.epsilon)
        self.reset()

    def reset(self):
        self.mcts.reset()

    def getActionProb(self, canonicalBoard, turn, temp = 1):
        return self.mcts.getActionProb(canonicalBoard, turn, temp)

    def getExpertProb(self, canonicalBoard, turn, temp=1, prune=False):
        return self.mcts.getExpertProb(canonicalBoard, turn, temp, prune)

    def processResults(self, pi, value):
        self.mcts.processResult(pi, value)

    def findLeafToProcess(self, canonicalBoard, turn, isRoot, isFast = False):
        noise = (np.zeros(self.action) + (1 / self.action)) if isFast else np.random.dirichlet([10 / self.action] * self.action)
        flag, ans = self.mcts.findLeafToProcess(canonicalBoard, noise, turn)
        return ans if flag else None
