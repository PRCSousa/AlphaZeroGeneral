import numpy as np
import math
from attax_game import Attaxx
from go_game import Go
#from matplotlib import pyplot
from copy import deepcopy

import torch
import torch.nn as nn
import torch.nn.functional as F

class Node():
    def __init__(self, game, C, state, player, parent = None, action_taken = None) -> None:
        self.game = game
        self.C = C
        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        self.player = player
        self.children = []
        self.expandable_moves = game.get_valid_moves(self.state, self.player)

        self.visit_count = 0
        self.value_sum = 0
    
    def is_fully_expanded(self):
        return len(self.expandable_moves) == 0 and len(self.children) > 0

    def select(self):
        best_child = None
        best_ucb = -np.inf

        for child in self.children:
            ucb  = self.get_ucb(child)
            if ucb > best_ucb:
                best_child = child
                best_ucb = ucb
        
        return best_child

    def get_ucb(self, child):
        q_value = 1 - ((child.value_sum/child.visit_count)+1) / 2
        return q_value + self.C * math.sqrt(math.log(self.visit_count) / child.visit_count)
    
    def expand(self, player):
        moves_arr = list(self.expandable_moves)
        random_index = np.random.choice(len(self.expandable_moves))
        action = moves_arr[random_index]
        self.expandable_moves.remove(action)
        #print("PLAYER NO EXPAND: " + str(player))

        child_state = deepcopy(self.state)

        child_state = self.game.get_next_state(child_state, action, player)

        child = Node(game = self.game, C = self.C, state = child_state, action_taken = action, player = -player, parent = self)

        self.children.append(child)
        #print("CHILDREN ammount on node " + str(self) + ": " + str(len(self.children)))
        return child
    
    def simulate(self, player):
        value, is_terminal = self.game.get_value_and_terminated(self.state)
        
        if is_terminal:
            return value

        rollout_state = deepcopy(self.state)
        rollout_player = player
        iter = 0
        while True:
            iter += 1
            # self.game.print_board(rollout_state)
            valid_moves = self.game.get_valid_moves(rollout_state, rollout_player)
            valid_moves = list(valid_moves)
            if len(valid_moves) == 0:
                value, is_terminal = self.game.get_value_and_terminated(rollout_state)
                return value
            action = np.random.choice(len(valid_moves))
            #print("Valid moves: " + str(valid_moves))
            action = valid_moves[action]
            rollout_state = self.game.get_next_state(rollout_state, action, rollout_player)
            value, is_terminal = self.game.get_value_and_terminated(rollout_state)
            if is_terminal or iter > 10:
                return value
            rollout_player = -rollout_player
    
    def backpropagate(self, value):
        self.value_sum += value
        self.visit_count += 1

        if self.parent is not None:
            self.parent.backpropagate(value)

class ResNet(nn.Module):
    def __init__(self, game, num_resBlocks, num_hidden):
        super().__init__()
        self.startBlock = nn.Sequential(
            nn.Conv2d(3, num_hidden, kernel_size=3, padding=1),
            nn.BatchNorm2d(num_hidden),
            nn.ReLU()
        )
        
        self.backBone = nn.ModuleList(
            [ResBlock(num_hidden) for i in range(num_resBlocks)]
        )
        
        self.policyHead = nn.Sequential(
            nn.Conv2d(num_hidden, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * game.x_dim * game.y_dim, game.action_size)
        )
        
        self.valueHead = nn.Sequential(
            nn.Conv2d(num_hidden, 3, kernel_size=3, padding=1),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3 * game.x_dim * game.y_dim, 1),
            nn.Tanh()
        )
        
    def forward(self, x):
        x = self.startBlock(x)
        for resBlock in self.backBone:
            x = resBlock(x)
        policy = self.policyHead(x)
        value = self.valueHead(x)
        return policy, value
    
class ResBlock(nn.Module):
    def __init__(self, num_hidden):
        super().__init__()
        self.conv1 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(num_hidden)
        self.conv2 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(num_hidden)
        
    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x

class MCTS():
    def __init__(self, game, args) -> None:
        self.game = game
        self.C = args['C']
        self.num_searches = args['num_searches']
    
    def search(self, state, player):
        #define root
        root = Node(self.game, self.C, state, player)
        #selection 
        for search in range(self.num_searches):
            node = root
            print("Search: " + str(search))
            # DEBUG
            #print("IS FULLY EXPANDED?: " + str(node.is_fully_expanded()))
            while node.is_fully_expanded():
                node = node.select()
            
            value, is_terminal = self.game.get_value_and_terminated(node.state)
            
            if not is_terminal:
                node = node.expand(player)
                value = node.simulate(player)
            
            node.backpropagate(value)

        action_probs = {}
        for child in root.children:
            action_probs[child.action_taken] = child.visit_count
        return action_probs

game = Attaxx([5, 5])

state = game.get_initial_state()
player = 1

state = game.get_initial_state()
state = game.get_next_state(state, [0, 0, 0, 2], 1)
state = game.get_next_state(state, [0, 4, 1, 4], -1)

print(state)
encoded_state = game.get_encoded_state(state)

print(encoded_state)

tensor_state = torch.tensor(encoded_state).unsqueeze(0)

model = ResNet(game, 4, 64)

policy, value = model(tensor_state)
value = value.item()
policy = torch.softmax(policy, axis=1).squeeze(0).detach().cpu().numpy()

print(value, policy)

import matplotlib.pyplot as plt

plt.bar(range(game.action_size), policy)
plt.show()