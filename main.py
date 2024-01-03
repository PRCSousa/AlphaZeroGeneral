import torch
from torch.optim import Adam
import random
import numpy as np
import math

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
            'num_selfPlay_iterations': 30,   # number of self-play games to play within each iteration
            'num_mcts_searches': 300,         # number of mcts simulations when selecting a move within self-play
            'max_moves': 512,                 # maximum number of moves in a game (to avoid infinite games which should not happen but just in case)
            'num_epochs': 400,                  # number of epochs for training on self-play data for each iteration
            'batch_size': 128,                # batch size for training
            'temperature': 1.30,              # temperature for the softmax selection of moves
            'C': 2,                           # the value of the constant policy
            'augment': True,                 # whether to augment the training data with flipped states
            'dirichlet_alpha': 0.03,           # the value of the dirichlet noise (alpha)
            'dirichlet_epsilon': 0.25,        # the value of the dirichlet noise (epsilon)
            'alias': ('Go' + SAVE_NAME)
        }

        game = Go()
        model = ResNet(game, 9, 3, device)
        optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

    elif GAME == 'Attaxx':
        size = int(input("Game size: (4/5/6) "))
        game_size = [size,size]
        args = {
            'game': 'Attaxx',
            'num_iterations': 200,             # number of highest level iterations
            'num_selfPlay_iterations': 50,   # number of self-play games to play within each iteration
            'num_mcts_searches': 50,         # number of mcts simulations when selecting a move within self-play
            'max_moves': 512,                 # maximum number of moves in a game (to avoid infinite games which should not happen but just in case)
            'num_epochs': 20,                 # number of epochs for training on self-play data for each iteration
            'batch_size': 16,                 # batch size for training
            'temperature': 1.25,              # temperature for the softmax selection of moves
            'C': 2,                           # the value of the constant policy
            'augment': False,                 # whether to augment the training data with flipped states
            'dirichlet_alpha': 0.5,           # the value of the dirichlet noise
            'dirichlet_epsilon': 0.125,       # the value of the dirichlet noise
            'alias': ('Attaxx' + SAVE_NAME)
        }

        game = Attaxx(game_size)
        model = ResNet(game, 20, 48, device)
        optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

    def print_array_as_grid_corrected(array):

        # Determine the size of the grid
        grid_size = int(math.sqrt(len(array) - 1))

        # Formatting each element to 4 decimal places
        formatted_array = [f"{elem:.4f}" for elem in array]

        # Printing the grid with improved alignment
        print("  " + " ".join([f"{i:6}" for i in range(grid_size)]))  # Column headers
        print("  +" + "-" * ((grid_size-1) * 8))

        for i in range(grid_size):
            row = formatted_array[i * grid_size:(i + 1) * grid_size]
            print(f"{i}| " + " ".join(row))  # Row with row header

        print(f"Skip Chance: {array[-1]}")

    if LOAD:
        model.load_state_dict(torch.load(f'AlphaZero/Models/{GAME+SAVE_NAME}/{MODEL}.pt', map_location=device))
        #model.load_state_dict(torch.load(f'AlphaZero/Models/{GAME+SAVE_NAME}/{MODEL}.pt', map_location=torch.device('cpu')))
        optimizer.load_state_dict(torch.load(f'AlphaZero/Models/{GAME+SAVE_NAME}/{OPT}.pt', map_location=device))

    if not TEST:
        os.makedirs(f'AlphaZero/Models/{GAME+SAVE_NAME}', exist_ok=True)
        alphaZero = AlphaZero(model, optimizer, game, args)
        alphaZero.learn()
    else:
        if not LOAD:
            print("No model to test")
            exit()

        PLAYER1 = input("Player 1: (user/AI) ")
        PLAYER2 = input("Player 2: (user/AI) ")
        print()

        if GAME == 'Go':
            game = Go()

            model.load_state_dict(torch.load(f'AlphaZero/Models/{GAME+SAVE_NAME}/{MODEL}.pt'))
            mcts = MCTS(model, game, args)
            state = game.get_initial_state()
            game.print_board(state)

            player = 1

            while True:
                if player == 1:
                    if PLAYER1 == 'user':
                        a, b = tuple(int(x.strip()) for x in input("\nInput your move: ").split(' '))
                        print("\n")
                        action = a * 9 + b
                        state = game.get_next_state(state, action, player)
                    else:
                        tmp_state = game.change_perspective(state, -1)
                        action = mcts.search(tmp_state, -player)                    
                        print_array_as_grid_corrected(action)
                        action = np.argmax(action)

                        print(f"\nAlphaZero Action: {action // game.row_count} {action % game.column_count}\n")
                        state = game.get_next_state(state, action, player)
                else:
                    if PLAYER2 == 'user':
                        a, b = tuple(int(x.strip()) for x in input("\nInput your move: ").split(' '))
                        print("\n")
                        action = a * 9 + b
                        state = game.get_next_state(state, action, player)
                    else:
                        action = mcts.search(state, player)                    
                        print_array_as_grid_corrected(action)
                        action = np.argmax(action)

                        print(f"\nAlphaZero Action: {action // game.row_count} {action % game.column_count}\n")
                        state = game.get_next_state(state, action, player)

                winner, win = game.get_value_and_terminated(state, action, player)
                if win:
                    game.print_board(state)
                    print(f"player {winner} wins")
                    exit()

                player = - player
                game.print_board(state)
            
        elif GAME == 'Attaxx':
            game = Attaxx(game_size)

            model.load_state_dict(torch.load(f'AlphaZero/Models/{GAME+SAVE_NAME}/{MODEL}.pt', map_location=device))
            mcts = MCTS(model, game, args)
            state = game.get_initial_state()
            game.print_board(state)

            player = 1

            while True:
                if player == 1:
                    if PLAYER1 == 'user':
                        move = tuple(int(x.strip()) for x in input("\nInput your move: ").split(' '))
                        print("\n")
                        action = game.move_to_int(move)
                        state = game.get_next_state(state, action, player)
                    else:
                        tmp_state = game.change_perspective(state, -1)
                        action = mcts.search(tmp_state, -player)
                        action = np.argmax(action)
                        print(f"\nAlphaZero Action: {game.int_to_move(action)}\n")
                        state = game.get_next_state(state, action, player)
                else:
                    if PLAYER2 == 'user':
                        move = tuple(int(x.strip()) for x in input("\nInput your move: ").split(' '))
                        print("\n")
                        action = game.move_to_int(move)
                        state = game.get_next_state(state, action, player)
                    else:
                        action = mcts.search(state, player)
                        action = np.argmax(action)
                        print(f"\nAlphaZero Action: {game.int_to_move(action)}\n")
                        state = game.get_next_state(state, action, player)

                winner, win = game.get_value_and_terminated(state, action, player)
                if win:
                    game.print_board(state)
                    print(f"player {winner} wins")
                    exit()

                player = -player
                game.print_board(state)
