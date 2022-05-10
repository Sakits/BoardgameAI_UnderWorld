import torch
from pathlib import Path
from glob import glob
from torch import multiprocessing as mp
from torch.utils.data import TensorDataset, ConcatDataset, DataLoader
from pytorch_classification.utils import Bar, AverageMeter
from queue import Empty
from time import time
import numpy as np
from math import ceil
import os
from tensorboardX import SummaryWriter

from mcts.MCTSWrapper import MCTSWrapper as MCTS
from train.SelfPlayAgent import SelfPlayAgent
from train.Arena import Arena
from game.Players import RandomPlayer, NNPlayer

class Coach:
    def __init__(self, game, nnet, args):
        np.random.seed()

        self.game = game
        self.nnet = nnet
        self.pnet = self.nnet.__class__(self.game)
        self.args = args

        networks = sorted(glob(self.args.checkpoint+'/*'))
        self.args.start_iter = len(networks)
        if self.args.start_iter == 0:
            self.nnet.save_checkpoint(
                folder=self.args.checkpoint, filename='iteration-0000.pkl')
            self.args.start_iter = 1

        self.nnet.load_checkpoint(
            folder=self.args.checkpoint, filename=f'iteration-{(self.args.start_iter-1):04d}.pkl')

        self.agents = []
        self.input_tensors = []
        self.policy_tensors = []
        self.value_tensors = []
        self.batch_ready = []
        self.ready_queue = mp.Queue()
        self.file_queue = mp.Queue()
        self.result_queue = mp.Queue()
        self.completed = mp.Value('i', 0)
        self.games_played = mp.Value('i', 0)
        if self.args.run_name != '':
            self.writer = SummaryWriter(log_dir='runs/'+self.args.run_name)
        else:
            self.writer = SummaryWriter()

    def learn(self):
        for i in range(self.args.start_iter, self.args.num_iters + 1):
            print(f'------ITER {i}------')
            self.generateSelfPlayAgents()
            self.processSelfPlayBatches()
            self.saveIterationSamples(i)
            self.processGameResults(i)
            self.killSelfPlayAgents()
            self.train(i)
            if self.args.compare_with_random and (i - 1) % self.args.random_compare_freq == 0:
                if i == 1:
                    print('Note: Comparisons with Random do not use monte carlo tree search.')
                self.compareToRandom(i)
            if self.args.compare_with_past and (i - 1) % self.args.past_compare_freq == 0:
                self.compareToPast(i)
            print()
        self.writer.close()

    def generateSelfPlayAgents(self):
        self.ready_queue = mp.Queue()
        feat_cnt, boardx, boardy = self.game.getFeatureSize()
        for i in range(self.args.workers):
            self.input_tensors.append(torch.zeros(
                [self.args.process_batch_size, feat_cnt, boardx, boardy]))
            self.input_tensors[i].pin_memory()
            self.input_tensors[i].share_memory_()

            self.policy_tensors.append(torch.zeros(
                [self.args.process_batch_size, self.game.getActionSize()]))
            self.policy_tensors[i].pin_memory()
            self.policy_tensors[i].share_memory_()

            self.value_tensors.append(torch.zeros(
                [self.args.process_batch_size, 1]))
            self.value_tensors[i].pin_memory()
            self.value_tensors[i].share_memory_()
            self.batch_ready.append(mp.Event())

            self.agents.append(
                SelfPlayAgent(i, self.game, self.ready_queue, self.batch_ready[i],
                              self.input_tensors[i], self.policy_tensors[i], self.value_tensors[i], self.file_queue,
                              self.result_queue, self.completed, self.games_played, self.args))
            self.agents[i].start()

    def processSelfPlayBatches(self):
        sample_time = AverageMeter()
        bar = Bar('Generating Samples', max=self.args.max_sample_num)
        end = time()

        n = 0
        while self.completed.value != self.args.workers:
            try:
                id = self.ready_queue.get(timeout=1)
                self.policy, self.value = self.nnet.process(
                    self.input_tensors[id])
                self.policy_tensors[id].copy_(self.policy)
                self.value_tensors[id].copy_(self.value)
                self.batch_ready[id].set()
            except Empty:
                pass

            if self.file_queue.qsize() > self.args.max_sample_num:
                lock = self.games_played.get_lock()
                lock.acquire()
                if self.games_played.value < self.args.max_games_per_iteration:
                    print("")
                    print(self.games_played.value, 'games played')
                self.games_played.value = self.args.max_games_per_iteration
                lock.release()

            size = self.file_queue.qsize()
            if size > n:
                sample_time.update((time() - end) / (size - n), size - n)
                n = size
                end = time()
            bar.suffix = f'({size}/{self.args.max_sample_num}) Sample Time: {sample_time.avg:.3f}s | Total: {bar.elapsed_td} | ETA: {bar.eta_td:}'
            bar.goto(size)
        bar.update()
        bar.finish()
        print()

    def killSelfPlayAgents(self):
        for i in range(self.args.workers):
            self.agents[i].join()
            del self.input_tensors[0]
            del self.policy_tensors[0]
            del self.value_tensors[0]
            del self.batch_ready[0]
        self.agents = []
        self.input_tensors = []
        self.policy_tensors = []
        self.value_tensors = []
        self.batch_ready = []
        self.ready_queue = mp.Queue()
        self.completed = mp.Value('i', 0)
        self.games_played = mp.Value('i', 0)

    def saveIterationSamples(self, iteration):
        num_samples = self.file_queue.qsize()
        print(f'Saving {num_samples} samples')
        feat_cnt, boardx, boardy = self.game.getFeatureSize()
        data_tensor = torch.zeros([num_samples, feat_cnt, boardx, boardy])
        policy_tensor = torch.zeros([num_samples, self.game.getActionSize()])
        value_tensor = torch.zeros([num_samples, 1])
        for i in range(num_samples):
            data, policy, value = self.file_queue.get()
            data_tensor[i] = torch.from_numpy(data)
            policy_tensor[i] = torch.tensor(policy)
            value_tensor[i, 0] = value

        os.makedirs(self.args.data, exist_ok=True)
        torch.save(
            data_tensor, f'{self.args.data}/iteration-{iteration:04d}-data.pkl')
        torch.save(policy_tensor,
                   f'{self.args.data}/iteration-{iteration:04d}-policy.pkl')
        torch.save(
            value_tensor, f'{self.args.data}/iteration-{iteration:04d}-value.pkl')

        del data_tensor
        del policy_tensor
        del value_tensor

    def processGameResults(self, iteration):
        num_games = self.result_queue.qsize()
        p1wins = 0
        p2wins = 0
        draws = 0
        for _ in range(num_games):
            winner = self.result_queue.get()
            if winner == 1:
                p1wins += 1
            elif winner == -1:
                p2wins += 1
            else:
                draws += 1

        print("p1_wins:", p1wins, " p2_wins:", p2wins, " draws:", draws)

        self.writer.add_scalar('win_rate/p1 vs p2',
                               (p1wins+0.5*draws)/num_games, iteration)
        self.writer.add_scalar('win_rate/draw', draws/num_games, iteration)

    def train(self, iteration):
        datasets = []
        currentHistorySize = min(max(4, (iteration + 4)//2),self.args.num_iters_for_train_examples_history)
        for i in range(max(1, iteration - currentHistorySize), iteration + 1):
            data_tensor = torch.load(
                f'{self.args.data}/iteration-{i:04d}-data.pkl')
            policy_tensor = torch.load(
                f'{self.args.data}/iteration-{i:04d}-policy.pkl')
            value_tensor = torch.load(
                f'{self.args.data}/iteration-{i:04d}-value.pkl')
            datasets.append(TensorDataset(
                data_tensor, policy_tensor, value_tensor))

        dataset = ConcatDataset(datasets)
        dataloader = DataLoader(dataset, batch_size=self.args.train_batch_size, shuffle=True,
                                num_workers=0, pin_memory=True)

        train_steps = min(self.args.train_steps_per_iteration, 
            2 * (iteration + 1 - max(1, iteration - currentHistorySize)) * self.args.max_sample_num // self.args.train_batch_size)
        l_pi, l_v = self.nnet.train(dataloader, train_steps)
        self.writer.add_scalar('loss/policy', l_pi, iteration)
        self.writer.add_scalar('loss/value', l_v, iteration)
        self.writer.add_scalar('loss/total', l_pi + l_v, iteration)

        self.nnet.save_checkpoint(
            folder=self.args.checkpoint, filename=f'iteration-{iteration:04d}.pkl')

        del dataloader
        del dataset
        del datasets

    def compareToPast(self, iteration):
        past = max(0, iteration-50)
        self.pnet.load_checkpoint(folder=self.args.checkpoint,
                                  filename=f'iteration-{past:04d}.pkl')
        print(f'PITTING AGAINST ITERATION {past}')
        if(self.args.arena_MCTS):
            pplayer = MCTS(self.game, self.pnet, self.args)
            nplayer = MCTS(self.game, self.nnet, self.args)

            def playpplayer(x, turn):
                if turn <= 2:
                    pplayer.reset()
                temp = self.args.temp if turn <= self.args.temp_threshold else self.args.arena_temp
                policy = pplayer.getActionProb(x, temp=temp)
                return np.random.choice(len(policy), p=policy)

            def playnplayer(x, turn):
                if turn <= 2:
                    nplayer.reset()
                temp = self.args.temp if turn <= self.args.temp_threshold else self.args.arena_temp
                policy = nplayer.getActionProb(x, temp=temp)
                return np.random.choice(len(policy), p=policy)

            arena = Arena(playnplayer, playpplayer, self.game)
        else:
            pplayer = NNPlayer(self.game, self.pnet, self.args.arena_temp)
            nplayer = NNPlayer(self.game, self.nnet, self.args.arena_temp)

            arena = Arena(nplayer.play, pplayer.play, self.game)
        nwins, pwins, draws = arena.playGames(self.args.arena_compare)

        print(f'NEW/PAST WINS : {nwins} / {pwins} ; DRAWS : {draws}\n')
        self.writer.add_scalar(
            'win_rate/to past', float(nwins + 0.5 * draws) / (pwins + nwins + draws), iteration)

    def compareToRandom(self, iteration):
        r = RandomPlayer(self.game)
        nnplayer = NNPlayer(self.game, self.nnet, self.args.arena_temp)
        print('PITTING AGAINST RANDOM')

        arena = Arena(nnplayer.play, r.play, self.game)
        nwins, pwins, draws = arena.playGames(self.args.arena_compare_random)

        print(f'NEW/RANDOM WINS : {nwins} / {pwins} ; DRAWS : {draws}\n')
        self.writer.add_scalar(
            'win_rate/to random', float(nwins + 0.5 * draws) / (pwins + nwins + draws), iteration)