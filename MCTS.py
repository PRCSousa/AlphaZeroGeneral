import numpy as np
import math
from attax_game import Attaxx
from go_game import Go
from matplotlib import pyplot
from copy import deepcopy

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

a = input("Attaxx or Go? (A/G): ")
if a == "A":
    mode = "Attaxx"
else:
    mode = "Go"

if mode == "Go":
    args = [5, 5, 5.5]  # x, y, komi
    attaxx_game = Go(args)
else:
    attaxx_game = Attaxx([5, 5])

args = {
    'C': 1.41,
    'num_searches': 10000
}

state = attaxx_game.get_initial_state()
mcts = MCTS(attaxx_game, args)
player = 1

while True: 
    attaxx_game.print_board(state)

    if player == 1:
        print("Player 1")
        if attaxx_game.check_available_moves(state, player):
            # print(attaxx_game.get_valid_moves(state, player))
            if mode == "Attaxx":
                a, b, a1, b1 = tuple(int(x.strip()) for x in input().split(' ')) #input e assim: 0 0 0 0
                action = (a, b, a1, b1)
            else:
                a, b = tuple(int(x.strip()) for x in input().split(' '))
                action = (a, b)
            if attaxx_game.is_valid_move(state, action, player):
                attaxx_game.get_next_state(state, action, player)
                player = - player
                winner, win = attaxx_game.check_win_and_over(state)
                if win:
                    attaxx_game.print_board(state)
                    print(f"player {winner} wins")
                    exit()
    
    else:
        print("Player -1 MCTS")
        print("BEGIN SEARCH")
        # print(attaxx_game.get_valid_moves(state, player))
        mcts_prob = mcts.search(state, player)
        action_selected = list(mcts_prob.keys())[0]
        print("END SEARCH")
        print("ACTION SELECTED: " + str(action_selected))
        for action in mcts_prob:
            print(str(action) + ": " + str(mcts_prob[action]))
            if mcts_prob[action] >= mcts_prob[action_selected]:
                #print("ACTION SELECTED: " + str(action))
                action_selected = action
        #print("ACTION SELECTED: " + str(action_selected))
        if attaxx_game.is_valid_move(state, action_selected, player):
                attaxx_game.get_next_state(state, action_selected, player)
                player = - player
                winner, win = attaxx_game.check_win_and_over(state)
                if win:
                    attaxx_game.print_board(state)
                    print(f"player {winner} wins")
                    exit()