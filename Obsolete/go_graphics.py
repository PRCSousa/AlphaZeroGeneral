import os
import math
import time
import json
import torch
import random
import pygame
import pickle
import socket
import numpy as np
import torch.nn as nn
from tqdm.notebook import trange
from torch.optim import Adam
import torch.nn.functional as F

def to_pixels(x):
    return SCREEN_PADDING + x*CELL_SIZE

def to_coord(x):
    quarter = CELL_SIZE//4
    closest = (x-SCREEN_PADDING)//CELL_SIZE
    if abs(to_pixels(closest)-(x-SCREEN_PADDING > to_pixels(closest)-(x-SCREEN_PADDING+quarter))):
        closest = (x-SCREEN_PADDING+quarter)//CELL_SIZE
    return closest

def draw_board():
    pygame.draw.rect(screen, GREEN, rect=(SCREEN_PADDING, SCREEN_PADDING, CELL_SIZE*(SIZE_BOARD-1), CELL_SIZE*(SIZE_BOARD-1)))
    for i in range(SIZE_BOARD):
        pygame.draw.line(screen, BLACK,(to_pixels(i),SCREEN_PADDING),(to_pixels(i),CELL_SIZE*(SIZE_BOARD-1) + SCREEN_PADDING),3)
        pygame.draw.line(screen, BLACK,(SCREEN_PADDING,to_pixels(i)),(CELL_SIZE*(SIZE_BOARD-1)+SCREEN_PADDING,to_pixels(i)),3)

def draw_piece(x,y,player):
    color = BLACK if player == -1 else WHITE
    pygame.draw.circle(screen,color,(to_pixels(x),to_pixels(y)),PIECE_SIZE)
    pygame.draw.circle(screen,BLACK,(to_pixels(x),to_pixels(y)),PIECE_SIZE,3)

def hover_to_select(player,valid_moves,click):
    mouse_x, mouse_y = pygame.mouse.get_pos()
    x, y = None, None
    if ([to_coord(mouse_x), to_coord(mouse_y)] in valid_moves):
        x, y = to_coord(mouse_x), to_coord(mouse_y)
    
    if (x!=None):
        pixels = (to_pixels(x),to_pixels(y))
        distance = pygame.math.Vector2(pixels[0] - mouse_x, pixels[1] - mouse_y).length()
        if distance < PIECE_SIZE:
            s = pygame.Surface((SCREEN_SIZE, SCREEN_SIZE), pygame.SRCALPHA)
            if player == 1:
                pygame.draw.circle(s,(255,255,255,200),(to_pixels(x),to_pixels(y)),PIECE_SIZE)
            if player == -1:
                pygame.draw.circle(s,(0,0,0,200),(to_pixels(x),to_pixels(y)),PIECE_SIZE)
            pygame.draw.circle(s,BLACK,(to_pixels(x),to_pixels(y)),PIECE_SIZE,3)
            screen.blit(s, (0, 0))

    return [None, None, player]


class Go():

    EMPTY = 0
    BLACK = 1
    WHITE = -1
    BLACKMARKER = 4
    WHITEMARKER = 5
    LIBERTY = 8

    def __init__(self, size, komi):
        self.row_count = size
        self.column_count = size
        self.komi = 5.5
        self.action_size = self.row_count * self.column_count + 1
        self.liberties = []
        self.block = []
        self.seki_liberties = []
        
    def get_initial_state(self):
        '''
        # Description:
        Returns a board of the argument size filled of zeros.

        # Retuns:
        Empty board full of zeros
        '''
        board = np.zeros((self.row_count, self.column_count))
        return board
    

    def count(self, x, y, state: list, player:int , liberties: list, block: list) -> tuple[list, list]:
        '''
        # Description:
        Counts the number of liberties of a stone and the number of stones in a block.
        Follows a recursive approach to count the liberties of a stone and the number of stones in a block.

        # Returns:
        A tuple containing the number of liberties and the number of stones in a block.
        '''
        
        #initialize piece
        piece = state[y][x]
        #if there's a stone at square of the given player
        if piece == player:
            #save stone coords
            block.append((y,x))
            #mark the stone
            if player == self.BLACK:
                state[y][x] = self.BLACKMARKER
            else:
                state[y][x] = self.WHITEMARKER
            
            #look for neighbours recursively
            if y-1 >= 0:
                liberties, block = self.count(x,y-1,state,player,liberties, block) #walk north
            if x+1 < self.column_count:
                liberties, block = self.count(x+1,y,state,player,liberties, block) #walk east
            if y+1 < self.row_count:
                liberties, block = self.count(x,y+1,state,player,liberties, block) #walk south
            if x-1 >= 0:
                liberties, block = self.count(x-1,y,state,player,liberties, block) #walk west

        #if square is empty
        elif piece == self.EMPTY:
            #mark liberty
            state[y][x] = self.LIBERTY
            #save liberties
            liberties.append((y,x))

        # print("Liberties: " + str(len(self.liberties)) + " in: " + str(x) + "," + str(y))
        # print("Block: " + str(len(self.block)) + " in: " + str(x) + "," + str(y))
        return liberties, block

    #remove captured stones
    def clear_block(self, block: list, state: list) -> list:
        '''
        # Description:
        Clears the block of stones captured by the opponent on the board.

        # Returns:
        The board with the captured stones removed.
        '''

        #clears the elements in the block of elements which is captured
        for i in range(len(block)): 
            y, x = block[i]
            state[y][x] = self.EMPTY
        
        return state

    #restore board after counting stones and liberties
    def restore_board(self, state: list) -> list:
        '''
        # Description:
        Restores the board to its original state after counting liberties and stones.
        This is done by unmarking the stones following bitwise operations with the global class variables.
        
        # Returns:
        The board with the stones unmarked.
        '''

        #unmark stones
        # print("Restore Board")
        # print(state)
        for y in range(len(state)):
            for x in range(len(state)):
                #restore piece
                val = state[y][x]
                if val == self.BLACKMARKER:
                    state[y][x] = self.BLACK
                elif val == self.WHITEMARKER:
                    state[y][x] = self.WHITE
                elif val == self.LIBERTY:
                    state[y][x] = self.EMPTY

        # print("After Restore Board")
        # print(state)
        return state

    def print_board(self, state: list) -> None:
            '''
            # Description:
            Draws the board in the console.

            # Returns:
            None
            '''

        # Print column coordinates
            print("   ", end="")
            for j in range(len(state[0])):
                print(f"{j:2}", end=" ")
            print("\n  +", end="")
            for _ in range(len(state[0])):
                print("---", end="")
            print()

            # Print rows with row coordinates
            for i in range(len(state)):
                print(f"{i:2}|", end=" ")
                for j in range(len(state[0])):
                    print(f"{str(int(state[i][j])):2}", end=" ")
                print()
    
    def captures(self, state: list,player: int, a:int, b:int) -> tuple[bool, list]:
        '''
        # Description:
        Checks if a move causes a capture of stones of the player passed as an argument.
        If a move causes a capture, the stones are removed from the board.

        # Returns:
        A tuple containing a boolean indicating if a capture has been made and the board with the captured stones removed.
        '''
        check = False
        neighbours = []
        if(a > 0): neighbours.append((a-1, b))
        if(a < self.column_count - 1): neighbours.append((a+1, b))
        if(b > 0): neighbours.append((a, b - 1))
        if(b < self.row_count - 1): neighbours.append((a, b+1))

        #loop over the board squares
        for pos in neighbours:
            # print(pos)
            x = pos[0]
            y = pos[1]    
            # init piece
            piece = state[x][y]

                #if stone belongs to given colour
            if piece == player:
                # print("opponent piece")
                # count liberties
                liberties = []
                block = []
                liberties, block = self.count(y, x, state, player, liberties, block)
                # print("Liberties in count: " + str(len(liberties)))
                # if no liberties remove the stones
                if len(liberties) == 0: 
                    #clear block
                    state = self.clear_block(block, state)
                    check = True

                #restore the board
                state = self.restore_board(state)

        #print("Captures: " + str(check))
        return check, state
    
    def set_stone(self, a, b, state, player):
        '''
        # Description:
        Places the piece on the board. THIS DOES NOT account for the rules of the game, use get_next_state().

        # Retuns:
        Board with the piece placed.
        '''
        state[a][b] = player
        return state
    
    def get_next_state(self, state, action, player):
        '''
        # Description
        Plays the move, verifies and undergoes captures and saves the state to the history.
        
        # Returns:
        New state with everything updated.
        '''
        if action == self.row_count * self.column_count:
            return state # pass move

        a = action // self.row_count
        b = action % self.column_count

        # checking if the move is part of is the secondary move to a ko fight
        state = self.set_stone(a, b, state, player)
        # print(state)
        state = self.captures(state, -player, a, b)[1]
        return state
    
    def is_valid_move(self, state: list, action: tuple, player: int) -> bool:
        '''
        # Description:
        Checks if a move is valid.
        If a move repeats a previous state or commits suicide (gets captured without capturing back), it is not valid.
        
        A print will follow explaining the invalid move in case it exists.

        # Returns:
        A boolean confirming the validity of the move.
        '''

        a = action[0]
        b = action[1]

        #print(f"{a} , {b}")

        statecopy = np.copy(state).astype(np.int8)

        if state[a][b] != self.EMPTY:
            # print("Space Occupied")
            return False 


        statecopy = self.set_stone(a,b,statecopy,player)

        if self.captures(statecopy, -player, a, b)[0] == True:
            return True
        else:
            #print("no captures")
            libs, block = self.count(b,a,statecopy,player,[],[])
            #print(libs)
            if len(libs) == 0:
                #print("Invalid, Suicide")
                return False
            else:
                return True
        

    def get_valid_moves(self, state, player):
        '''
        # Description:
        Returns a matrix with the valid moves for the current player.
        '''
        newstate = np.zeros((self.row_count, self.column_count))
        for a in range(0, self.column_count):
            for b in range(0, self.row_count):
                if self.is_valid_move(state, (a,b), player):
                    newstate[a][b] = 1
        
        newstate = newstate.reshape(-1)

        empty = 0
        endgame = True
        
        for x in range(self.column_count):
            for y in range(self.row_count):
                if state[x][y] == self.EMPTY:
                    empty += 1
                    if empty >= self.column_count * self.row_count // 3: # if 2/3ds are already filled, skipping becomes available
                        endgame = False
                        break
        if endgame:
            newstate = np.concatenate([newstate, [1]])
        else:
            newstate = np.concatenate([newstate, [0]])
        return (newstate).astype(np.int8)

    def get_value_and_terminated(self, state, action, player):
        '''
        # Description:
        Returns the value of the state and if the game is over.
        '''

        scoring, endgame = self.scoring(state)

        if endgame:
            if player == self.BLACK:
                if scoring > 0:
                    return 1, True
                else:
                    return -1, True
            else:
                if scoring < 0:
                    return 1, True
                else:
                    return -1, True
        else:
            if player == self.BLACK:
                if scoring > 0:
                    return 1, False
                else:
                    return -1, False
            else:
                if scoring < 0:
                    return 1, False
                else:
                    return -1, False


        
    def scoring(self, state: list) -> int:
        '''
        # Description:
        Checks the score of the game. Score is calculated using:

        black - (white + komi)

        # Returns:
        Integer with score.
        '''
        black = 0
        white = 0
        empty = 0
        endgame = True

        for x in range(self.column_count):
            for y in range(self.row_count):
                if state[x][y] == self.EMPTY:
                    empty += 1
                    if empty >= self.column_count * self.row_count // 4:
                        endgame = False
                        break

        black, white = self.count_influenced_territory_enhanced(state)
        black_eyes, black_strong_groups = self.count_eyes_and_strong_groups(state, self.BLACK)
        white_eyes, white_strong_groups = self.count_eyes_and_strong_groups(state, self.WHITE)
        # print(f"Black | Territory: {black} Eyes: {black_eyes} Strong Groups: {black_strong_groups}")
        # print(f"White | Territory: {white} Eyes: {white_eyes} Strong Groups: {white_strong_groups}")
        
        black += black_eyes + black_strong_groups
        white += white_eyes + white_strong_groups
        
        return black - (white + self.komi), endgame
    
    def count_influenced_territory_enhanced(self, board: list) -> tuple[int, int]:
        '''
        # Description 
        Calculates the territory influenced by black and white players on the Go board.

        This function iterates through the board, analyzing each empty point to determine 
        if it's influenced by the surrounding black or white stones. The influence is calculated
        based on the adjacent stones, with positive scores indicating black influence and negative
        scores indicating white influence.

        # Returns:
        Tuple (black_territory, white_territory)
        '''
        black_territory = 0
        white_territory = 0
        visited = set()

        # Function to calculate influence score
        def influence_score(x, y):
            score = 0
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nx, ny = x + dx, y + dy
                if 0 <= nx < len(board) and 0 <= ny < len(board[0]):
                    score += board[nx][ny]
            return score

        # Function to explore territory
        def explore_territory(x, y):
            nonlocal black_territory, white_territory
            if (x, y) in visited or not (0 <= x < len(board) and 0 <= y < len(board[0])):
                return
            visited.add((x, y))

            if board[x][y] == 0:
                score = influence_score(x, y)
                if score > 0:
                    black_territory += 1
                elif score < 0:
                    white_territory += 1

        for i in range(len(board)):
            for j in range(len(board[0])):
                if board[i][j] == 0 and (i, j) not in visited:
                    explore_territory(i, j)

        return black_territory, white_territory
    
    def is_eye(self, board, x, y, player):

        # An eye is an empty point with all adjacent points of the player's color
        # and at least one diagonal point of the player's color.
        
        if board[x][y] != self.EMPTY:
            return False
        
        for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
            nx, ny = x + dx, y + dy
            if not (0 <= nx < len(board) and 0 <= ny < len(board[0])):
                continue
            if board[nx][ny] != player:
                return False
            
        true_eye = False
        count = 0
        for dx, dy in [(1, 1), (1, -1), (-1, 1), (-1, -1)]:
            nx, ny = x + dx, y + dy

            if 0 <= nx < len(board) and 0 <= ny < len(board[0]) and board[nx][ny] == player:
                count += 1
                if count >= 3:
                    true_eye = True


        return true_eye

    def count_eyes_and_strong_groups(self, board, player):
        eyes = 0
        strong_groups = 0
        visited = set()

        def dfs(x, y):
            if (x, y) in visited or board[x][y] != player:
                return 0

            visited.add((x, y))
            liberties = 0
            for dx, dy in [(1, 0), (-1, 0), (0, 1), (0, -1)]:
                nx, ny = x + dx, y + dy
                if not (0 <= nx < len(board) and 0 <= ny < len(board[0])):
                    continue
                if board[nx][ny] == self.EMPTY:
                    liberties += 1
                elif board[nx][ny] == player:
                    liberties += dfs(nx, ny)

            return liberties

        for x in range(len(board)):
            for y in range(len(board[0])):
                if board[x][y] == player and (x, y) not in visited:
                    liberties = dfs(x, y)
                    if liberties >= 2:  # Arbitrary threshold for a strong group
                        strong_groups += 1
                if board[x][y] != player and (x, y) not in visited and self.is_eye(board, x, y, player):
                    eyes += 1

        return eyes, strong_groups


    def get_opponent(self, player):
        '''
        # Description:
        Changes Opponent
        '''
        return -player
    
    def get_opponent_value(self, value):
        '''
        # Description
        Returns the negative value of the value
        '''
        return -value
    
    def get_encoded_state(self, state):
        '''
        # Description: 
        Converts the current state of the Go board into a 3-layer encoded format suitable for neural network input.
        Each layer in the encoded format represents the presence of a specific type of stone or an empty space on the board:
        - Layer 1 encodes the positions of white stones (represented by -1 in the input state) as 1s, and all other positions as 0s.
        - Layer 2 encodes the positions of empty spaces (represented by 0 in the input state) as 1s, and all other positions as 0s.
        - Layer 3 encodes the positions of black stones (represented by 1 in the input state) as 1s, and all other positions as 0s.
        This encoding helps in clearly distinguishing between different elements on the board for machine learning applications.

        # Returns: 
        A NumPy array of shape (3, height, width) containing the 3-layer encoded representation of the board state. Each layer is a 2D array where the board's height and width correspond to the dimensions of the original state.
        '''
        layer_1 = np.where(np.array(state) == -1, 1, 0).astype(np.float32)
        layer_2 = np.where(np.array(state) == 0, 1, 0).astype(np.float32)
        layer_3 = np.where(np.array(state) == 1, 1, 0).astype(np.float32)

        result = np.stack([layer_1, layer_2, layer_3]).astype(np.float32)

        return result
    
    def change_perspective(self, state, player):
        '''
        # Description: 
        Adjusts the perspective of the Go board state based on the current player.

        # Returns: 
        A two-dimensional array representing the Go board state adjusted for the current player's perspective.
        '''
        return state * player

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
        '''
        # Description: 
        Selects the best child node from the current node's children in a Monte Carlo Tree Search using the Upper Confidence Bound (UCB) algorithm. 

        # Returns: 
        The best child node, chosen based on the highest UCB value or randomly if there's a tie.
        '''
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
        '''
        # Description: 
        Calculates the Upper Confidence Bound (UCB) value for a given child node in a Monte Carlo Tree Search.

        # Returns: 
        The calculated UCB value for the given child node.
        '''
        if child.visit_count == 0:
            q_value = child.prior * self.args['C'] * (math.sqrt(self.visit_count)) / (child.visit_count + 1)
        else:
            q_value = -(child.value_sum / child.visit_count) + child.prior * self.args['C'] * (math.sqrt(self.visit_count)) / (child.visit_count + 1)
        return q_value

    def serialize(self):
        # Serialize only essential data
        node_data = {
            'game': self.game,
            'args': self.args,
            'parent': self.parent,
            'state': self.state,
            'action_taken': self.action_taken,
            'player': self.player,
            'prior': self.prior,
            'visit_count': self.visit_count,
            'value_sum': self.value_sum,
            'children': [child for child in self.children]  # Assuming each child has a unique ID
        }
        return json.dumps(node_data)


    def deserialize(node_json):
        # Convert JSON back into a Node object
        node_data = json.loads(node_json)
        node = Node(  # assuming constructor can handle this data
            game = node_data['game'],
            args = node_data['args'],
            parent = node_data['parent'],
            player = node_data['player'],
            state=node_data['state'],
            action_taken=node_data['action_taken'],
            prior=node_data['prior'],
            visit_count=node_data['visit_count'],
        )
        node.value_sum = node_data['value_sum']

        for child in node_data['children']:
            child.parent = node
            node.children.append(child)

        # You'll need to handle children reconstruction separately
        return node
    
    def expand(self, policy):
        '''
        # Description: 
        Expands the current node by adding new child nodes based on the given policy probabilities. For each possible action, it calculates the next state, adjusts the perspective for the opponent, and creates a new child node if the probability for that action is greater than zero.

        # Returns: 
        None
        '''
        for action, prob in enumerate(policy):
            if prob > 0:
                child_state = self.state.copy()
                child_state = self.game.get_next_state(child_state, action, 1)
                child_state = self.game.change_perspective(child_state, player=-1)
                child = Node(self.game, self.args, child_state, self.game.get_opponent(self.player), self, action, prob)
                self.children.append(child)
            
    def backpropagate(self, value):
        '''
        # Description: 
        Performs the backpropagation step in Monte Carlo Tree Search. It updates the current node's value sum and visit count based on the received value.

        # Returns: 
        None
        '''
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
        self.tree_dict = {}
        
    @torch.no_grad()
    def search(self, states, player):
        """
        # Description:
        Performs Monte Carlo Tree Search (MCTS) in batch to find the best action.

        # Returns:
        An array of arrays of action probabilities for each possible action.
        """

        action_prob_list = []

        for state in states:
            
            root = Node(self.game, self.args, state, player, visit_count=1)

            searches = self.args['num_mcts_searches']

            if str(state)+str(player) in self.tree_dict.keys():
                action_prob_list.append(self.tree_dict.get(str(state)+str(player)))
                continue
            
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
                
            for search in range(searches):
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
            action_prob_list.append(action_probs)

            self.tree_dict.update({str(state)+str(player): action_probs})

        return action_prob_list

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
        # Description:
        The forward pass of the model. This overrides the forward method of nn.Module so that it can be called directly on the model.

        # Returns:
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
    # Description:
    A residual block for the ResNet model.
    '''
    def __init__(self, num_hidden):
        super().__init__()
        self.conv1 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding="same")
        self.bn1 = nn.BatchNorm2d(num_hidden)
        self.conv2 = nn.Conv2d(num_hidden, num_hidden, kernel_size=3, padding="same")
        self.bn2 = nn.BatchNorm2d(num_hidden)
        
    def forward(self, x):
        """
        # Description:
        Forward pass through the residual block.

        # Returns:
        Output tensor after passing through the block.
        """
        residual = x
        x = F.relu(self.bn1(self.conv1(x)))
        x = self.bn2(self.conv2(x))
        x += residual
        x = F.relu(x)
        return x

class AlphaZero:
    def __init__(self, model, optimizer, game, args):
        self.model = model
        self.optimizer = optimizer
        self.game = game
        self.args = args
        self.mcts = MCTS(model, game, args)

    def augment_state(self, state, probs):

        augmented_states = []

        skip_prob = probs[-1]
        action_probs_matrix = np.array(probs[:-1]).reshape(self.game.column_count, self.game.row_count)
        augmented_action_probs = []

        def augment_and_append(transformed_state, transformed_probs_matrix):

            # Append state
            augmented_states.append(transformed_state)

            # Flatten probs matrix, append the last value, and then append to augmented_action_probs
            augmented_action_probs.append(list(transformed_probs_matrix.flatten()) + [skip_prob])

        # Original state and probs
        augment_and_append(state, action_probs_matrix)

        # Rotate 90 degrees clockwise
        augment_and_append(np.rot90(state, k=1), np.rot90(action_probs_matrix, k=1))

        # Rotate 180 degrees clockwise
        augment_and_append(np.rot90(state, k=2), np.rot90(action_probs_matrix, k=2))

        # Rotate 270 degrees clockwise
        augment_and_append(np.rot90(state, k=3), np.rot90(action_probs_matrix, k=3))

        # Flip horizontally
        augment_and_append(np.fliplr(state), np.fliplr(action_probs_matrix))

        # Flip vertically
        augment_and_append(np.flipud(state), np.flipud(action_probs_matrix))

        # Rotate 90 degrees clockwise and flip horizontally
        augment_and_append(np.rot90(np.fliplr(state), k=1), np.rot90(np.fliplr(action_probs_matrix), k=1))

        # Rotate 90 degrees clockwise and flip vertically
        augment_and_append(np.rot90(np.flipud(state), k=1), np.rot90(np.flipud(action_probs_matrix), k=1))

        return augmented_states, augmented_action_probs


    def selfPlay(self):
        player = 1

        memory = []
        states = []

        for _ in range(0, self.args['parallel_games']):
            state = self.game.get_initial_state()
            states.append(state)
            memory.append([])

        iter = 0
        prev_skip = False
        temperature = self.args['temperature']
        debugging = False

        returnData = []

        while True:
            if self.args["game"] == "Attaxx" and debugging:
                print("\nSEARCHING...")

            neutral_states_list = []

            for state in states:
                neutral_states_list.append(self.game.change_perspective(state, player))

            action_probs_list = self.mcts.search(states, player)

            for i, (neutral_state, action_probs) in enumerate(zip(neutral_states_list, action_probs_list)):
                memory[i].append((neutral_state, action_probs, player))

            for idx, (state, action_probs) in enumerate(zip(states, action_probs_list)):
                temperature_action_probs = action_probs ** (1 / temperature)
                temperature_action_probs /= np.sum(temperature_action_probs)

                action = np.random.choice(self.game.action_size, p=temperature_action_probs)

                state = self.game.get_next_state(state, action, player)

                if self.args["game"] == "Attaxx" and debugging:
                    print(f"Player: {player} with move {self.game.int_to_move(action)}\nBoard:")
                    self.game.print_board(state)    

                value, is_terminal = self.game.get_value_and_terminated(state, action, player)
                    

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
                    for hist_neutral_state, hist_action_probs, hist_player in memory[idx]:
                        hist_outcome = value if hist_player == player else self.game.get_opponent_value(value)

                        if self.args['augment']:
                            augmented_states, augmented_action_probs = self.augment_state(hist_neutral_state, hist_action_probs)

                            for augmented_state, augmented_probs in zip(augmented_states, augmented_action_probs):
                                returnMemory.append((self.game.get_encoded_state(augmented_state), augmented_probs, hist_outcome))
                        else:
                            returnMemory.append((self.game.get_encoded_state(hist_neutral_state), hist_action_probs, hist_outcome))

                        returnData = returnData + returnMemory

                    del memory[idx]
                    del states[idx]

                if len(memory) <= 0:
                    return returnData

            player = self.game.get_opponent(player)

            if temperature >= 0.1:
                temperature = temperature * self.args['cooling_constant']
            else:
                temperature = 0.1

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
    
    def learn(self, memory = None, LAST_ITERATION=0):
        primary_memory = []

        if memory != None:
            primary_memory = memory

        for iteration in range(LAST_ITERATION+1, self.args['num_iterations']):
            print(f"Iteration {iteration + 1}")

            secondary_memory = []

            for selfPlay_iteration in trange(self.args['num_selfPlay_iterations']):
                states = self.selfPlay()
                self.mcts.tree_dict = {}
                secondary_memory += states

            training_memory = []
            if self.args['experience_replay']:
                sample_size = int(len(primary_memory) * 0.3)

                training_memory += random.sample(primary_memory, min(sample_size, len(primary_memory)))
                training_memory += secondary_memory
                
                primary_memory += secondary_memory
            else:
                training_memory += secondary_memory

            print(f"Memory size: {len(training_memory)}")

            self.model.train()

            for epoch in trange(self.args['num_epochs']):
                self.train(training_memory)

            print("\n")
                
            torch.save(self.model.state_dict(), f"DevelopmentModels/{self.args['alias']}/model_{iteration}.pt")
            torch.save(self.optimizer.state_dict(), f"DevelopmentModels/{self.args['alias']}/optimizer_{iteration}.pt")
            with open(f'DevelopmentModels/{self.args["alias"]}/memory_{iteration}.pkl', 'wb') as f:
                pickle.dump(primary_memory, f)
            print("Data Saved!")


torch.manual_seed(0)
random.seed(0)
np.random.seed(0)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


SAVE_NAME = None

if __name__ == '__main__':

    # Go / Attaxx
    GAME = "Go"

    # Board size (7/9 for Go, 4/5/6 for Attaxx)
    SIZE = 7

    # True to load previous model
    # False to start from scratch
    LOAD = True
    LAST_ITERATION = 1

    # Save Name
    SAVE_NAME = "7x7Parallel"

    # False for training
    # True for playing
    TEST = True

    if GAME == 'Go':
        if SIZE == 7:
            args = {
                'game': 'Go',
                'num_iterations': 20,             # number of highest level iterations
                'num_selfPlay_iterations': 15,    # number of self-play games to play within each iteration
                'num_mcts_searches': 200,         # number of mcts simulations when selecting a move within self-play
                'max_moves': 512,                 # maximum number of moves in a game (to avoid infinite games which should not happen but just in case)
                'num_epochs': 20,                 # number of epochs for training on self-play data for each iteration
                'batch_size': 16,                 # batch size for training
                'temperature': 3,                 # temperature for the softmax selection of moves
                'cooling_constant': 0.90,         # value that gets multiplied to the temperature to gradually reduce it  
                'C': 2,                           # the value of the constant policy
                'experience_replay': True,        # recycle a certain % of old random selfplay data in the current training iteration
                'augment': False,                 # whether to augment the training data with flipped and rotated states
                'parallel_games': 10,            # number of games run in parallel
                'dirichlet_alpha': 0.03,          # the value of the dirichlet noise (alpha)
                'dirichlet_epsilon': 0.25,        # the value of the dirichlet noise (epsilon)
                'alias': ('Go' + SAVE_NAME)
            }

            game = Go(size = SIZE, komi = 5.5)
            model = ResNet(game, 10, 10, device)
            optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)
            
        elif SIZE == 9:
            args = {
                'game': 'Go',
                'num_iterations': 20,             # number of highest level iterations
                'num_selfPlay_iterations': 20,    # number of self-play games to play within each iteration
                'num_mcts_searches': 200,         # number of mcts simulations when selecting a move within self-play
                'max_moves': 512,                 # maximum number of moves in a game (to avoid infinite games which should not happen but just in case)
                'num_epochs': 60,                 # number of epochs for training on self-play data for each iteration
                'batch_size': 32,                 # batch size for training
                'temperature': 3,                 # temperature for the softmax selection of moves
                'cooling_constant': 0.85,         # value that gets multiplied to the temperature to gradually reduce it  
                'C': 2,                           # the value of the constant policy
                'experience_replay': True,        # recycle a certain % of old random selfplay data in the current training iteration
                'augment': False,                 # whether to augment the training data with flipped and rotated states
                'parallel_games': 5,            # number of games run in parallel
                'dirichlet_alpha': 0.032,          # the value of the dirichlet noise (alpha)
                'dirichlet_epsilon': 0.25,        # the value of the dirichlet noise (epsilon)
                'alias': ('Go' + SAVE_NAME)
            }

            game = Go(size = SIZE, komi = 5.5)
            model = ResNet(game, 15, 15, device)
            optimizer = Adam(model.parameters(), lr=0.001, weight_decay=0.0001)

    if LOAD:
        model.load_state_dict(torch.load(f'DevelopmentModels/{GAME+SAVE_NAME}/model_{LAST_ITERATION}.pt', map_location=device))
        optimizer.load_state_dict(torch.load(f'DevelopmentModels/{GAME+SAVE_NAME}/optimizer_{LAST_ITERATION}.pt', map_location=device))

        if GAME == 'Go':
            PLAYER1 = "AI"
            PLAYER2 = "AI"
            game = Go(SIZE, 5.5)

            model.load_state_dict(torch.load(f'DevelopmentModels/{GAME+SAVE_NAME}/model_{LAST_ITERATION}.pt', map_location = device))
            mcts = MCTS(model, game, args)
            state = game.get_initial_state()
            #game.print_board(state)

            player = 1
            prev_skip = False

    
            rendering = True

            SIZE_BOARD = SIZE
            BLACK = (0,0,0)
            WHITE = (255,255,255)
            GREEN = (188,106,66)
            SCREEN_SIZE = 600
            SCREEN_PADDING = 50
            CELL_SIZE = (SCREEN_SIZE - SCREEN_PADDING) // SIZE_BOARD
            PIECE_SIZE = (SCREEN_SIZE - 2*SCREEN_PADDING) // SIZE_BOARD // 3

            click = False
            valid_moves = []
            for i in range(SIZE_BOARD):
                for j in range(SIZE_BOARD):
                    valid_moves.append([i, j])

            cur_pieces = []

            a = None

            if not rendering:
                game.print_board(state)
            else:
                pygame.init()
                pygame_icon = pygame.image.load('image.png')
                pygame.display.set_icon(pygame_icon)

                screen=pygame.display.set_mode((SCREEN_SIZE,SCREEN_SIZE))

                pygame.display.set_caption("Go")

            while True:

                if rendering:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            pygame.quit()
                        if event.type == pygame.MOUSEBUTTONDOWN:
                            click = True
                            a, b, player = hover_to_select(player, valid_moves, click)
                        if event.type == pygame.MOUSEBUTTONUP:
                            click = False

                    screen.fill(GREEN)
                    draw_board()
                    for i in range(0,len(state)):
                        for j in range(0,len(state)):
                            if state[i][j] == 0:
                                if game.is_valid_move(state, (i,j), player):
                                    valid_moves.append([i,j])
                            else:
                                draw_piece(i,j, state[i][j])

                if player == 1:
                    
                    if PLAYER1 == 'user':

                        valid_move_selected = False

                        a, b, player = hover_to_select(player, valid_moves, click)

                        if click:

                            mouse_x, mouse_y = pygame.mouse.get_pos()
                            a, b = to_coord(mouse_x), to_coord(mouse_y)

                            action = a * SIZE + b

                            state = game.get_next_state(state, action, player)

                            winner, win = game.get_value_and_terminated(state, action, player)
                
                            if action == game.action_size:
                                if prev_skip:
                                    win = True
                                else:
                                    prev_skip = True
                            else:
                                prev_skip = False

                            if win:
                                print(f"player {winner} wins")
                                break

                            player = -player

                    else:
                        tmp_state = game.change_perspective(state, -1)
                        action = mcts.search([tmp_state], -player)                    
                        action = np.argmax(action[0])
                        print(f"\nAlphaZero Action: {action // game.row_count} {action % game.column_count}\n")
                        state = game.get_next_state(state, action, player)

                        winner, win = game.get_value_and_terminated(state, action, player)
                
                        if action == game.action_size:
                            if prev_skip:
                                win = True
                            else:
                                prev_skip = True
                        else:
                            prev_skip = False

                        if win:
                            print(f"player {winner} wins")
                            break

                        player = -player
                else:
                    if PLAYER2 == 'user':
                        valid_move_selected = False
                        a, b, player = hover_to_select(player, valid_moves, click)
                        if click:
                            mouse_x, mouse_y = pygame.mouse.get_pos()
                            a, b = to_coord(mouse_x), to_coord(mouse_y)
                            action = a * SIZE + b
                            state = game.get_next_state(state, action, player)
                            winner, win = game.get_value_and_terminated(state, action, player)
                
                            if action == game.action_size:
                                if prev_skip:
                                    win = True
                                else:
                                    prev_skip = True
                            else:
                                prev_skip = False

                            if win:
                                print(f"player {winner} wins")
                                break

                            player = -player
                    else:
                        action = mcts.search([state], player)                    
                        action = np.argmax(action[0])
                        

                        print(f"\nAlphaZero Action: {action // game.row_count} {action % game.column_count}\n")
                        state = game.get_next_state(state, action, player)

                        winner, win = game.get_value_and_terminated(state, action, player)
                
                        if action == game.action_size:
                            if prev_skip:
                                win = True
                            else:
                                prev_skip = True
                        else:
                            prev_skip = False

                        if win:
                            print(f"player {winner} wins")
                            break

                        player = -player

                pygame.display.flip()