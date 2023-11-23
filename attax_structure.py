import numpy as np

class Attaxx:
    def __init__(self, *args):
        self.x_dim = args[0]
        self.y_dim = args[1]
    
    def get_initial_state(self):
        return np.zeros((self.x_dim, self.y_dim))

    def get_next_state(self, state, action, player):
        a, b, a1, b1 = action
        

    def is_valid_move(self, state, action, player):
        
        a, b, a1, b1 = action
        
        if abs(a-b)>2 or abs(a1-b1)>2:
            return False

        return True

        

    def check_win():
        pass

    def get_value_and_terminated(self):
        pass