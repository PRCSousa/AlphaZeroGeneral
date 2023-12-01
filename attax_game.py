import numpy as np

class Attaxx:
    def __init__(self, args):
        self.x_dim = args[0]
        self.y_dim = args[1]
    
    def get_initial_state(self):
        state = np.zeros((self.x_dim, self.y_dim))
        state[0][0] = 1
        state[self.x_dim-1][self.y_dim-1] = 1
        state[0][self.x_dim-1] = -1
        state[self.y_dim-1][0] = -1
        return state

    def get_next_state(self, state, action, player):
        a, b, a1, b1 = action
        if abs(a-a1)==2 or abs(b-b1)==2:
            state[a][b] = 0
            state[a1][b1] = player
        else:
            state[a1][b1] = player
        self.capture_pieces(state, action, player)
        

    def is_valid_move(self, state, action, player):
        a, b, a1, b1 = action
        if abs(a-a1)>2 or abs(b-b1)>2 or state[a1][b1]!=0 or state[a][b]!=player or ((abs(a-a1)==1 and abs(b-b1)==2) or (abs(a-a1)==2 and abs(b-b1)==1)):
            return False

        return True

    def capture_pieces(self, state, action, player):
        a, b, a1, b1 = action
        for i in range(a1-1, a1+2):
            for j in range(b1-1, b1+2):
                try:
                    if state[i][j]==-player and i>=0 and j>=0:
                        state[i][j] = player
                except IndexError:
                    pass
                continue

    def check_available_moves(self, state, player):
        for i in range(self.x_dim):
            for j in range(self.y_dim):
                if state[i][j] == player:
                    for a in range(self.x_dim):
                        for b in range(self.y_dim):
                            action = (i, j, a, b)
                            if self.is_valid_move(state, action, player):
                                return True
        return False

    def get_valid_moves(self, state, player):

        possible_moves = set()

        for i in range(self.x_dim):
            for j in range(self.y_dim):
                if state[i][j] == player:
                    moves_at_point = set(self.get_moves_at_point(state, player, i, j))
                    possible_moves = possible_moves.union(moves_at_point)
        
        return possible_moves

    def get_moves_at_point(self, state, player, a, b):

        moves_at_point = []

        for i in range(self.x_dim):
            for j in range(self.y_dim):
                possible_action = (a, b, i, j)
                if self.is_valid_move(state, possible_action, player):
                    moves_at_point.append(possible_action)
        return moves_at_point 

    def check_board_full(self, state):
        for row in state:
            if 0 in row:
                return False
        
        return True

    def check_win_and_over(self, state):

        count_player1 = 0
        count_player2 = 0

        for i in range(self.x_dim):
            for j in range(self.y_dim):
                if state[i][j] == 1:
                    count_player1+=1
                elif state[i][j] == -1:
                    count_player2+=1
        if count_player1 == 0:
            return True, -1
        elif count_player2 == 0:
            return True, 1
        
        if self.check_board_full(state):
            if count_player1>count_player2:
                return True, 1
            elif count_player2>count_player1:
                return True, -1
            elif count_player1==count_player2:
                return True, 2
        
        return False, 0
    
    def get_value_and_terminated(self, state, player):
        game_over, winner = self.check_win_and_over()
        return game_over, winner
    
    def print_board(self, state):
        state = state.astype(int)
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
                print(f"{str(state[i][j]):2}", end=" ")
            print()

attaxx_game = Attaxx([5, 5])
state = attaxx_game.get_initial_state()
player = 1

game = True

while game:

    print(f"Player {player} to move")
    attaxx_game.print_board(state)


    if attaxx_game.check_available_moves(state, player):
        a, b, a1, b1 = tuple(int(x.strip()) for x in input().split(' ')) #input e assim: 0 0 0 0
        action = (a, b, a1, b1)
        if attaxx_game.is_valid_move(state, action, player):
            print(attaxx_game.get_valid_moves(state, player))
            attaxx_game.get_next_state(state, action, player)
            player = - player
            win, winner = attaxx_game.check_win_and_over(state)
            if win:
                attaxx_game.print_board(state)
                print(f"player {winner} wins")
                exit()
    else:
        player = -player