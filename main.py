from torch import multiprocessing as mp

from train.Coach import Coach
from network.NNetWrapper import NNetWrapper as nn
from utils import *
# select game
from libcpp import Game

args = dotdict({
    'run_name': 'game_data',
    'workers': 4,
    'start_iter': 1,
    'num_iters': 1000,
    'process_batch_size': 128,
    'train_batch_size': 512,
    'train_steps_per_iteration': 1000,
    # Training Net may be slow if the value of max_sample_num is too small
    'max_sample_num': 10000, 
    # should be large enough
    'max_games_per_iteration': 100000,
    'num_iters_for_train_examples_history': 100,
    'symmetric_samples': False,
    # Dirichlet noise (should be small when num_MCTS_sims is small)
    'epsilon' : 0.25,
    # should be larger than action size to get a better result
    'num_MCTS_sims': 800,
    'num_fast_sims': 200,
    'prob_fast_sim': 0.75,
    'temp_threshold': 10,
    'temp': 1,
    'compare_with_random': True,
    'arena_compare_random': 500,
    'arena_compare': 100,
    'arena_temp': 1,
    'arena_MCTS': False,
    'random_compare_freq': 1,
    'compare_with_past': True,
    'past_compare_freq': 1,
    'cpuct': 2.5,
    'checkpoint': 'checkpoint',
    'data': 'data',
})

if __name__ == "__main__":
    g = Game(9, 5)
    nnet = nn(g)
    c = Coach(g, nnet, args)
    c.learn()
