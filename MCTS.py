import numpy as np
import math

class Node():
    def __init__(self, game, args, state, parent = None, action_takin = None) -> None:
        self.game = game
        self.args = args
        self.state = state
        self.parent = parent
        self.action_taken = action_takin

        self.children = []
        self.expandable_moves = game.get_valid_moves(state)

        self.visit_count = 0
        self.value_sum = 0
    
    def is_fully_expanded(self):
        return np.sum(self.expandable_moves) == 0 and len(self.children) > 0

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
        return q_value + self.args['C'] * math.sqrt(math.log(self.visit_count) / child.visit_count)

class MCTS():
    def __init__(self, game, args, state, parent = None, action_takin = None) -> None:
        self.game = game
        self.args = args
    
    def search(self, state):
        #define root
        root = Node(self.game, self.args, state)

        #selection 
        for search in range(self.args['num_searches']):
            node = root

            while node.is_fully_expanded():
                node = node.select()
            
            value, is_terminal = self.game.get_value_and_terminated(node.state, node.player)
            
            #expansion
            #simulation
            #backpropagation
            pass

