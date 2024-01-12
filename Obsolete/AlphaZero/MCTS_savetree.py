class Node:
    '''
    # Alpha Zero Node
    ## Description:
        A node for the AlphaZero MCTS. It contains the state, the action taken to get to the state, the prior probability of the action, the visit count, the value sum, and the children of the node.
    ## Methods:
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
        self.tree_cache = {}  # store tree in dict

    @torch.no_grad()
    def search(self, states, player):
        """
        # Description:
        Performs Monte Carlo Tree Search (MCTS) in batch to find the best action.

        # Returns:
        An array of arrays of action probabilities for each possible action.
        """
        roots = []

        for state in states:
            state_key = tuple(map(tuple, state))  # otherwise unhashable type error
            if state_key in self.tree_cache: #check if search tree already exists and reuse
                root = self.tree_cache[state_key]
            else:
                #otherwise create new
                root = Node(self.game, self.args, state, player, visit_count=1)
                self.tree_cache[state_key] = root

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
                        is_terminal = True  # if the action is pass when the previous action was also pass, end the game

                if not is_terminal:
                    policy, value = self.model(
                        torch.tensor(self.game.get_encoded_state(node.state), device=self.model.device).unsqueeze(0))
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

        action_prob_list = []
        for root in roots:
            action_probs = np.zeros(self.game.action_size)
            for child in root.children:
                action_probs[child.action_taken] = child.visit_count
            action_probs /= np.sum(action_probs)
            action_prob_list.append(action_probs)
        return action_prob_list


