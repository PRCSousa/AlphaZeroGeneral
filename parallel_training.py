import torch
from torch.optim import Adam
import random
import numpy as np

from Go.go import Go
from Attaxx.attaxx import Attaxx

from AlphaZero.alphazero_parallel import AlphaZeroParallel
from AlphaZero.alphazero_parallel import MCTSParallel
from AlphaZero.alphazero_parallel import ResNetParallel

import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

game = Attaxx([5, 5])

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ResNetParallel(game, 9, 128, device)

optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

args = {
            'game': 'Go',
            'num_iterations': 20,             # number of highest level iterations
            'num_selfPlay_iterations': 10,   # number of self-play games to play within each iteration
            'num_parallel_games': 100,        # number of games played in parallel
            'num_mcts_searches': 100,         # number of mcts simulations when selecting a move within self-play
            'num_epochs': 200,                  # number of epochs for training on self-play data for each iteration
            'batch_size': 8,                # batch size for training
            'temperature': 1.25,              # temperature for the softmax selection of moves
            'C': 2,                           # the value of the constant policy
            'augment': False,                 # whether to augment the training data with flipped states
            'dirichlet_alpha': 0.3,           # the value of the dirichlet noise
            'dirichlet_epsilon': 0.25,        # the value of the dirichlet noise
            'alias': ('Go')
        }

alphaZero = AlphaZeroParallel(model, optimizer, game, args)
alphaZero.learn()