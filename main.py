from torch import multiprocessing as mp

from train.Coach import Coach
from network.NNetWrapper import NNetWrapper as nn
from utils import *
# select game
from libcpp import Game

args = dotdict({
    'run_name': 'game_data',
    'workers': mp.cpu_count(),
    'start_iter': 1,
    'num_iters': 1000,
    'process_batch_size': 64,
    'train_batch_size': 512,
    'train_steps_per_iteration': 1000,
    # Training Net may be slow if the value of max_sample_num is too small
    'max_sample_num': 30000, 
    # should be large enough
    'max_games_per_iteration': 100000,
    'num_iters_for_train_examples_history': 50,
    'symmetric_samples': False,
    'num_MCTS_sims': 50,
    'num_fast_sims': 5,
    'prob_fast_sim': 0.75,
    'temp_threshold': 10,
    'temp': 1,
    'compare_with_random': True,
    'arena_compare_random': 500,
    'arena_compare': 500,
    'arena_temp': 0.1,
    'arena_MCTS': False,
    'random_compare_freq': 5,
    'compare_with_past': True,
    'past_compare_freq': 5,
    'cpuct': 3,
    'checkpoint': 'checkpoint',
    'data': 'data',
})

if __name__ == "__main__":
    g = Game(9, 5)
    nnet = nn(g)
    c = Coach(g, nnet, args)
    c.learn()
