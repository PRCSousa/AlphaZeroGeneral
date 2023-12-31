import numpy as np
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import os
from tqdm import trange
import pickle


class ResNet(nn.Module):
    '''
    # ResNet
    ## Description:
        A ResNet model for AlphaZero.
        The model takes in a state and outputs a policy and value.
         - The policy is a probability distribution over all possible actions.
         - The value is a number between -1 and 1, where -1 means the current player loses and 1 means the current player wins following a tanh activation.
        '''
    def __init__(self, game, num_resBlocks, num_hidden, device):
        super().__init__()
        self.device = device

        self.startBlock = nn.Sequential(
            nn.Conv2d(3, num_hidden, kernel_size=3, padding="same"),
            nn.BatchNorm2d(num_hidden),
            nn.ReLU()
        )
        self.backBone = nn.ModuleList(
            [ResBlock(num_hidden) for i in range(num_resBlocks)]
        )
        self.policyHead = nn.Sequential(
            nn.Conv2d(num_hidden, 32, kernel_size=3, padding="same"),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * game.row_count * game.column_count, game.action_size)
        )
        self.valueHead = nn.Sequential(
            nn.Conv2d(num_hidden, 3, kernel_size=3, padding="same"),
            nn.BatchNorm2d(3),
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(3 * game.row_count * game.column_count, 1),
            nn.Tanh()
        )
        
        self.to(device)
        
    def forward(self, x):
        '''
        ## Description:
            The forward pass of the model. This overrides the forward method of nn.Module so that it can be called directly on the model.
            ## Parameters:
            - `x`: The input tensor.
            ## Returns:
            - `policy`: The policy output of the model.
            - `value`: The value output of the model.
            '''
        x = self.startBlock(x)
        for resBlock in self.backBone:
            x = resBlock(x)
        policy = self.policyHead(x)
        value = self.valueHead(x)
        return policy, value
        
class ResBlock(nn.Module):
    '''
    # ResBlock
    ## Description:
        A residual block for the ResNet model.
    '''
    def __init__(self, num_hidden):
        super().__init__()
        self.conv1 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding="same")
        self.bn1 = nn.BatchNorm2d(num_hidden)
        self.conv2 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding="same")
        self.bn2 = nn.BatchNorm2d(num_hidden)
        
    def forward(self, x):
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x

class Node:
    '''
    # Alpha Zero Node
    ## Description:
        A node for the AlphaZero MCTS. It contains the state, the action taken to get to the state, the prior probability of the action, the visit count, the value sum, and the children of the node.
    ## Metohds:
        - `is_expanded()`: Returns whether the node has been expanded.
        - `select()`: Selects the best child node based on the UCB.
        - `get_ucb()`: Returns the UCB of a child node.
        - `expand()`: Expands the node by adding children.
        - `backpropagate()`: Backpropagates the value of the node to the parent node.
        '''
    def __init__(self, game, args, state, player, parent=None, action_taken=None, prior=0, visit_count=0):
        self.game = game
        self.args = args
        self.state = state
        self.parent = parent
        self.action_taken = action_taken
        self.player = player
        self.prior = prior
        self.children = []
        
        self.visit_count = visit_count
        self.value_sum = 0
        
    def is_expanded(self):
        '''
        # is_expanded
        ## Description:
            Returns whether the node has been expanded.
        ## Returns:
            - `bool`: Whether the node has been expanded.'''
        return len(self.children) > 0
    
    def select(self):
        best_child = []
        best_ucb = -np.inf
        
        for child in self.children:
            ucb = self.get_ucb(child)
            if ucb > best_ucb:
                best_child = [child]
                best_ucb = ucb
            elif ucb == best_ucb:
                best_child.append(child)
                
        return best_child[0] if len(best_child) == 1 else random.choice(best_child)
    
    def get_ucb(self, child):
        if child.visit_count == 0:
            q_value = 0
        else:
            q_value = 1 - ((child.value_sum / child.visit_count) + 1) / 2
        return q_value + self.args['C'] * (math.sqrt(self.visit_count) / (child.visit_count + 1)) * child.prior
    
    def expand(self, policy):
        for action, prob in enumerate(policy):
            if prob > 0:
                child_state = self.state.copy()
                child_state = self.game.get_next_state(child_state, action, 1)
                child_state = self.game.change_perspective(child_state, player=-1)
                child = Node(self.game, self.args, child_state, self.game.get_opponent(self.player), self, action, prob)
                self.children.append(child)
            
    def backpropagate(self, value):
        self.value_sum += value
        self.visit_count += 1
        
        if self.parent is not None:
            value = self.game.get_opponent_value(value)
            self.parent.backpropagate(value)  

class MCTS:
    def __init__(self, model, game, args):
        self.model = model
        self.game = game
        self.args = args
        
    @torch.no_grad()
    def search(self, state, player):
        root = Node(self.game, self.args, state, player, visit_count=1)
        
        policy, _ = self.model(
            torch.tensor(self.game.get_encoded_state(state), device=self.model.device).unsqueeze(0)
        )
        policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()
        policy = (1 - self.args['dirichlet_epsilon']) * policy + self.args['dirichlet_epsilon'] \
            * np.random.dirichlet([self.args['dirichlet_alpha']] * self.game.action_size)
        
        valid_moves = self.game.get_valid_moves(state, player)

        if self.args["game"] == "Attaxx":
            if np.sum(valid_moves) == 0:
                valid_moves[-1] = 1
            else:
                valid_moves[-1] = 0

        policy *= valid_moves
        policy /= np.sum(policy)
        root.expand(policy)
        
        for search in range(self.args['num_mcts_searches']):
            node = root
            while node.is_expanded():
                node = node.select()
            
            value, is_terminal = self.game.get_value_and_terminated(node.state, node.action_taken, node.player)
            value = self.game.get_opponent_value(value)
            
            if node.parent is not None:
                if node.action_taken == self.game.action_size - 1 and node.parent.action_taken == self.game.action_size - 1 and self.args['game'] == 'Go':
                    is_terminal = True # if the action is pass when the previous action was also pass, end the game

            if not is_terminal:
                policy, value = self.model(torch.tensor(self.game.get_encoded_state(node.state), device=self.model.device).unsqueeze(0))
                policy = torch.softmax(policy, axis=1).squeeze(0).cpu().numpy()
                valid_moves = self.game.get_valid_moves(node.state, player)

                if self.args["game"] == "Attaxx":
                    if np.sum(valid_moves) == 0:
                        valid_moves[-1] = 1
                    else:
                        valid_moves[-1] = 0

                policy *= valid_moves
                policy /= np.sum(policy)
                
                value = value.item()
                node.expand(policy)

            node.backpropagate(value)    
            
        action_probs = np.zeros(self.game.action_size)
        for child in root.children:
            action_probs[child.action_taken] = child.visit_count
        action_probs /= np.sum(action_probs)
        return action_probs

class AlphaZero:
    def __init__(self, model, optimizer, game, args):
        self.model = model
        self.optimizer = optimizer
        self.game = game
        self.args = args
        self.mcts = MCTS(model, game, args)

    def augment_state(self, state):

        augmented_states = []
        
        # Original state
        augmented_states.append(state)
        
        # Rotate 90 degrees clockwise
        augmented_states.append(np.rot90(state, k=1))
        
        # Rotate 180 degrees clockwise
        augmented_states.append(np.rot90(state, k=2))
        
        # Rotate 270 degrees clockwise
        augmented_states.append(np.rot90(state, k=3))
        
        # Flip horizontally
        augmented_states.append(np.fliplr(state))
        
        # Flip vertically
        augmented_states.append(np.flipud(state))
        
        # Rotate 90 degrees clockwise and flip horizontally
        augmented_states.append(np.rot90(np.fliplr(state), k=1))
        
        # Rotate 90 degrees clockwise and flip vertically
        augmented_states.append(np.rot90(np.flipud(state), k=1))
        
        return augmented_states
        
    def selfPlay(self):
        memory = []
        player = 1
        state = self.game.get_initial_state()
        iter = 0
        prev_skip = False

        debugging = False

        while True:
            if self.args["game"] == "Attaxx" and debugging:
                print("\nSEARCHING...")
            neutral_state = self.game.change_perspective(state, player)
            action_probs = self.mcts.search(state, player)
            memory.append((neutral_state, action_probs, player))

            temperature_action_probs = action_probs ** (1 / self.args['temperature'])
            temperature_action_probs /= np.sum(temperature_action_probs)
            action = np.random.choice(self.game.action_size, p=temperature_action_probs)

            if self.args["game"] == "Go":
                print(f"\nPlayer: {player}")
                if action != self.game.action_size - 1:
                    print(f"Action: {action // self.game.row_count} {action % self.game.column_count}")
                else:
                    print(f"Action: Skip")
                
            state = self.game.get_next_state(state, action, player)

            if self.args["game"] == "Attaxx" and debugging:
                print(f"Player: {player} with move {self.game.int_to_move(action)}\nBoard:")
                self.game.print_board(state)    

            value, is_terminal = self.game.get_value_and_terminated(state, action, player)

            if self.args["game"] == "Go":
                self.game.print_board(state)
                print(f"Evaluation: {value}")
                b, w = self.game.count_influenced_territory_enhanced(state)
                print(f"Influence: B:{b} W: {w}")
                

            if action == self.game.action_size - 1 and self.args['game'] == 'Go':
                if prev_skip:
                    is_terminal = True
                else:
                    prev_skip = True
            else:
                prev_skip = False

            if is_terminal or iter >= self.args['max_moves']:
                returnMemory = []
                if self.args["game"] == "Attaxx" and debugging:
                    print("GAME OVER\n\n")
                for hist_neutral_state, hist_action_probs, hist_player in memory:
                    hist_outcome = value if hist_player == player else self.game.get_opponent_value(value)
                    augmented_states = self.augment_state(hist_neutral_state)

                    for augmented_state in augmented_states:
                        returnMemory.append(
                            (self.game.get_encoded_state(augmented_state), hist_action_probs, hist_outcome)
                        )
                return returnMemory

            player = self.game.get_opponent(player)
            iter += 1
                
    def train(self, memory):
        random.shuffle(memory)
        for batchIdx in range(0, len(memory), self.args['batch_size']):
            sample = memory[batchIdx:batchIdx+self.args['batch_size']]
            state, policy_targets, value_targets = zip(*sample)
            
            state, policy_targets, value_targets = np.array(state), np.array(policy_targets), np.array(value_targets).reshape(-1, 1)
            
            state = torch.tensor(state, dtype=torch.float32, device=self.model.device)
            policy_targets = torch.tensor(policy_targets, dtype=torch.float32, device=self.model.device)
            value_targets = torch.tensor(value_targets, dtype=torch.float32, device=self.model.device)
            
            out_policy, out_value = self.model(state)
            
            policy_loss = F.cross_entropy(out_policy, policy_targets)
            value_loss = F.mse_loss(out_value, value_targets)
            loss = policy_loss + value_loss
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
    
    def learn(self):
        primary_memory = []

        for iteration in range(self.args['num_iterations']):
            print(f"Iteration {iteration + 1}")

            secondary_memory = []

            for selfPlay_iteration in trange(self.args['num_selfPlay_iterations']):
                states = self.selfPlay()
                secondary_memory += states

            training_memory = []

            sample_size = int(len(primary_memory) * 0.3)
            training_memory += random.sample(primary_memory, min(sample_size, len(primary_memory)))

            training_memory += secondary_memory
            primary_memory += secondary_memory

            print(f"Memory size: {len(training_memory)}")

            self.model.train()

            for epoch in trange(self.args['num_epochs']):
                self.train(training_memory)

            print("\n")
                
            torch.save(self.model.state_dict(), f"AlphaZero/Models/{self.args['alias']}/model_{iteration}.pt")
            torch.save(self.optimizer.state_dict(), f"AlphaZero/Models/{self.args['alias']}/optimizer_{iteration}.pt")