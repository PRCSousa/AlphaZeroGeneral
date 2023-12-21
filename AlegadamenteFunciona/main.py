import torch
from torch.optim import Adam
import random
import numpy as np
from go import Go
# from attax_game import Attaxx
from alphaZeroParallel import ResNet
from alphaZeroParallel import AlphaZero
import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

GAME = 'Go'
LOAD = False
PARALLEL = True

if __name__ == '__main__':
    if GAME == 'Go':
        args = {
            'game': 'Go',
            'num_iterations': 5,             # number of highest level iterations
            'num_selfPlay_iterations': 4,   # number of self-play games to play within each iteration
            'num_parallel_games': 100,        # number of games to play in parallel
            'num_mcts_searches': 40,         # number of mcts simulations when selecting a move within self-play
            'num_epochs': 4,                  # number of epochs for training on self-play data for each iteration
            'batch_size': 128,                # batch size for training
            'temperature': 1.25,              # temperature for the softmax selection of moves
            'C': 2,                           # the value of the constant policy
            'augment': False,                 # whether to augment the training data with flipped states
            'dirichlet_alpha': 0.3,           # the value of the dirichlet noise
            'dirichlet_epsilon': 0.25,        # the value of the dirichlet noise
        }

        game = Go()
        model = ResNet(game, 9, 3, device)
        optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

    elif GAME == 'Attaxx':
            args = {
                'game': 'Attaxx',
                'num_iterations': 8,              # number of highest level iterations
                'num_selfPlay_iterations': 4,   # number of self-play games to play within each iteration
                'num_parallel_games': 100,        # number of games to play in parallel
                'num_mcts_searches': 60,          # number of mcts simulations when selecting a move within self-play
                'num_epochs': 4,                  # number of epochs for training on self-play data for each iteration
                'batch_size': 64,                 # batch size for training
                'temperature': 1.25,              # temperature for the softmax selection of moves
                'C': 2,                           # the value of the constant policy
                'augment': False,                 # whether to augment the training data with flipped states
                'dirichlet_alpha': 0.3,           # the value of the dirichlet noise
                'dirichlet_epsilon': 0.125,       # the value of the dirichlet noise
            }

            # game = Attaxx()
            # model = ResNet(game, 9, 128, device)
            # optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

    if LOAD:
        model.load_state_dict(torch.load(f'Models/{game}/model.pt', map_location=device))
        optimizer.load_state_dict(torch.load(f'Models/{game}/optimizer.pt', map_location=device))
    else:
        if not os.path.exists(f'Models/{GAME}'):
            os.mkdir(f'Models/{GAME}')

    alphaZero = AlphaZero(model, optimizer, game, args)
    alphaZero.learn()