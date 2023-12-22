import torch
from torch.optim import Adam
import random
import numpy as np

from Go.go import Go
from Attaxx.attaxx import Attaxx

from AlphaZero.alphaZero import ResNet
from AlphaZero.alphaZero import AlphaZero
from AlphaZero.alphaZero import MCTS

import os

os.chdir(os.path.dirname(os.path.abspath(__file__)))

torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

SAVE_NAME = None

if __name__ == '__main__':

    GAME = input("Game: (Go/Attaxx) ")

    LOAD = input("Load:\nTrue will load a previous model, False will start from scratch (True/False):\n")
    if LOAD == 'True':
        LOAD = True
        SAVE_NAME = input("Alias of the model: ")
        MODEL = input("Model name: ")
        OPT = input("Optimizer name: ")
    else:
        LOAD = False
        SAVE_NAME = input("Alias of the new model: ")

    TEST = input("Test:\nTrue will play against the model, False will train the model (True/False):\n")
    if TEST == 'True':
        TEST = True	
    else:
        TEST = False

    if GAME == 'Go':
        args = {
            'game': 'Go',
            'num_iterations': 20,             # number of highest level iterations
            'num_selfPlay_iterations': 100,   # number of self-play games to play within each iteration
            'num_parallel_games': 100,        # number of games to play in parallel
            'num_mcts_searches': 60,         # number of mcts simulations when selecting a move within self-play
            'num_epochs': 10,                  # number of epochs for training on self-play data for each iteration
            'batch_size': 128,                # batch size for training
            'temperature': 1.25,              # temperature for the softmax selection of moves
            'C': 2,                           # the value of the constant policy
            'augment': False,                 # whether to augment the training data with flipped states
            'dirichlet_alpha': 0.3,           # the value of the dirichlet noise
            'dirichlet_epsilon': 0.25,        # the value of the dirichlet noise
            'alias': ('Go' + SAVE_NAME)
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
            'alias': ('Attaxx' + SAVE_NAME)
        }

        # game = Attaxx()
        # model = ResNet(game, 9, 128, device)
        # optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

    if LOAD:
        model.load_state_dict(torch.load(f'AlphaZero/Models/{GAME+SAVE_NAME}/{MODEL}.pt', map_location=device))
        optimizer.load_state_dict(torch.load(f'AlphaZero/Models/{GAME+SAVE_NAME}/{OPT}.pt', map_location=device))

    if not TEST:
        os.makedirs(f'AlphaZero/Models/{GAME+SAVE_NAME}', exist_ok=True)
        alphaZero = AlphaZero(model, optimizer, game, args)
        alphaZero.learn()
    else:
        if not LOAD:
            print("No model to test")
            exit()
        if GAME == 'Go':
            game = Go()

            model.load_state_dict(torch.load(f'AlphaZero/Models/{GAME+SAVE_NAME}/{MODEL}.pt'))
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
            
        elif GAME == 'Attaxx':
            game = Attaxx()
            name = input("Model File Name: ")

            model.load_state_dict(torch.load('AlphaZero/Models/Attaxx/' + name + '.pt'))

            mcts = MCTS(model, game, args)
            state = game.get_initial_state()
            game.print_board(state)

            player = 1

            while True:
                if player == 1:
                    
                    # input do player

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