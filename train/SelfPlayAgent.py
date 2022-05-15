import numpy as np
import torch
import torch.multiprocessing as mp

from mcts.MCTSWrapper import MCTSWrapper as MCTS

class SelfPlayAgent(mp.Process):

    def __init__(self, id, game, ready_queue, batch_ready, batch_tensor, policy_tensor, value_tensor, output_queue,
             result_queue, complete_count, games_played, args):
        super().__init__()
        self.id = id
        self.game = game
        self.ready_queue = ready_queue
        self.batch_ready = batch_ready
        self.batch_tensor = batch_tensor
        self.batch_size = self.batch_tensor.shape[0]
        self.policy_tensor = policy_tensor
        self.value_tensor = value_tensor
        self.output_queue = output_queue
        self.result_queue = result_queue
        self.games = []
        self.canonical = []
        self.histories = []
        self.player = []
        self.turn = []
        self.mcts = []
        self.games_played = games_played
        self.complete_count = complete_count
        self.args = args
        self.valid = torch.zeros_like(self.policy_tensor)
        self.fast = False
        for _ in range(self.batch_size):
            self.games.append(self.game.getInitBoard())
            self.histories.append([])
            self.player.append(1)
            self.turn.append(1)
            self.mcts.append(MCTS(self.game, None, self.args))
            self.canonical.append(None)

    def run(self):
        np.random.seed()
        while self.games_played.value < self.args.max_games_per_iteration:
            self.generateCanonical()
            self.fast = np.random.random_sample() < self.args.prob_fast_sim
            if self.fast:
                for i in range(self.args.num_fast_sims):
                    self.generateBatch()
                    self.processBatch()
            else:
                for i in range(self.args.num_MCTS_sims):
                    self.generateBatch()
                    self.processBatch()
            self.playMoves()
        with self.complete_count.get_lock():
            self.complete_count.value += 1
        self.output_queue.close()
        self.output_queue.join_thread()

    def generateBatch(self):
        for i in range(self.batch_size):
            board = self.mcts[i].findLeafToProcess(self.canonical[i], True, isFast = self.fast)
            if board is not None:
                self.batch_tensor[i] = torch.from_numpy(board)
        self.ready_queue.put(self.id)

    def processBatch(self):
        self.batch_ready.wait()
        self.batch_ready.clear()
        for i in range(self.batch_size):
            self.mcts[i].processResults(
                self.policy_tensor[i].data.numpy(), self.value_tensor[i][0])

    def playMoves(self):
        for i in range(self.batch_size):
            temp = int(self.turn[i] < self.args.temp_threshold and np.random.rand() > 0.5)
            if temp != 0:
                decay = self.args.temp / self.args.temp_threshold
                temp = temp - decay * self.turn[i]
            policy = self.mcts[i].getExpertProb(
                self.canonical[i], temp, not self.fast)
            action = np.random.choice(len(policy), p=policy)
            if not self.fast:
                self.histories[i].append((self.game.getFeature(self.canonical[i]), 
                    self.mcts[i].getExpertProb(self.canonical[i], prune=True), self.player[i]))
            self.games[i], self.player[i] = self.game.getNextState(self.games[i], self.player[i], action)
            self.turn[i] += 1

            # self.game.display(self.canonical[i])
            # print(temp)
            # for x in range(9):
            #     for y in range(9):
            #         print(policy[x * 9 + y], end = ' ')
            #     print('\n')

            winner = self.game.getGameEnded(self.games[i], 1)
            if winner != 0:
                self.result_queue.put(winner)
                lock = self.games_played.get_lock()
                lock.acquire()
                if self.games_played.value < self.args.max_games_per_iteration:
                    self.games_played.value += 1
                    lock.release()
                    for hist in self.histories[i]:
                        if self.args.symmetric_samples:
                            sym = self.game.getSymmetries(hist[0], hist[1])
                            for b, p in sym:
                                self.output_queue.put((b, p, winner * hist[2]))
                        else:
                            self.output_queue.put((hist[0], hist[1], winner * hist[2]))
                    self.games[i] = self.game.getInitBoard()
                    self.histories[i] = []
                    self.player[i] = 1
                    self.turn[i] = 1
                    self.mcts[i] = MCTS(self.game, None, self.args)
                else:
                    lock.release()

    def generateCanonical(self):
        for i in range(self.batch_size):
            self.canonical[i] = self.game.getCanonicalForm(
                self.games[i], self.player[i])
