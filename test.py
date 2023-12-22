from go import Go
# from attax_game import Attaxx
from alphaZeroParallel import ResNet
from alphaZeroParallel import AlphaZero
from alphaZeroParallel import MCTS
import os
import torch
from torch.optim import Adam
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
model.load_state_dict(torch.load('Models/Go/model_4.pt'))
optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
alphaZero = AlphaZero(model, optimizer, game, args)
mcts = MCTS(model, game, args)
state = game.get_initial_state()
game.print_board(state)
player = 1
while True:
    if player == 1:
        a, b = tuple(int(x.strip()) for x in input().split(' '))
        action = a * 5 + b
        state = game.get_next_state(state, action, player)
    else:
        neut = game.change_perspective(state, player)
        action = mcts.search(neut)
        action = np.argmax(action)
        state = game.get_next_state(state, action, player)

    winner, win = game.get_value_and_terminated(state, action)
    if win:
        game.print_board(state)
        print(f"player {winner} wins")
        exit()

    player = - player
    game.print_board(state)

