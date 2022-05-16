from pathlib import Path
import pprint
from glob import glob
from utils import *
from network.NNetWrapper import NNetWrapper as nn
# from othello.special.NNetSpecialWrapper import NNetSpecialWrapper as nns
from libcpp import Game
from tensorboardX import SummaryWriter
from game.Players import *
from mcts.MCTSWrapper import MCTSWrapper as MCTS
from train.Arena import Arena
import numpy as np
# import choix

"""
use this script to play every x agents against a single agent and graph win rate.
"""

args = dotdict({
    'run_name': 'pit-multi',
    'arena_compare': 100,
    'arena_temp': 0,
    'temp': 1,
    'temp_threshold': 10,
    # use zero if no montecarlo
    'num_MCTS_sims': 50,
    'cpuct': 1.5,
    'epsilon': 0.25, 
    'x': 10,
})

if __name__ == '__main__':
    print('Args:')
    pprint.pprint(args)
    benchmark_agent = "checkpoint/iteration-0264.pkl"
    
    if args.run_name != '':
        writer = SummaryWriter(log_dir='runs/'+args.run_name)
    else:
        writer = SummaryWriter()
    if not Path('checkpoint').exists():
        Path('checkpoint').mkdir()
    print('Beginning comparison')
    networks = sorted(glob('checkpoint/*'))
    temp = networks[::args['x']]
    if temp[-1] != networks[-1]:
        temp.append(networks[-1])
    
    networks = temp
    model_count = len(networks)

    if model_count < 1:
        print(
            "Too few models for pit multi.")
        exit()

    total_games = model_count * args.arena_compare
    print(
        f'Comparing {model_count} different models in {total_games} total games')

    g = Game(9, 5)
    nnet1 = nn(g)
    nnet2 = nn(g)

    nnet1.load_checkpoint(folder="", filename=benchmark_agent)
    short_name = Path(benchmark_agent).stem

    if args.num_MCTS_sims <= 0:
        p1 = NNPlayer(g, nnet1, args.arena_temp).play
    else:
        mcts1 = MCTS(g, nnet1, args)

        def p1(x, turn):
            if turn <= 2:
                mcts1.reset()
            temp = args.temp if turn <= args.temp_threshold else args.arena_temp
            policy = mcts1.getActionProb(x, temp=temp)
            return np.random.choice(len(policy), p=policy)
    
    for i in range(model_count):
        file = Path(networks[i])
        print(f'{short_name} vs {file.stem}')

        nnet2.load_checkpoint(folder='checkpoint', filename=file.name)
        if args.num_MCTS_sims <= 0:
            p2 = NNPlayer(g, nnet2, args.arena_temp).play
        else:
            mcts2 = MCTS(g, nnet2, args)

            def p2(x, turn):
                if turn <= 2:
                    mcts2.reset()
                temp = args.temp if turn <= args.temp_threshold else args.arena_temp
                policy = mcts2.getActionProb(x, temp=temp)
                return np.random.choice(len(policy), p=policy)

        arena = Arena(p1, p2, g)
        p1wins, p2wins, draws = arena.playGames(args.arena_compare)
        writer.add_scalar(
            f'Win Rate vs {short_name}', (p2wins + 0.5*draws)/args.arena_compare, i*args.x)
        print(f'wins: {p1wins}, ties: {draws}, losses:{p2wins}\n')
    writer.close()