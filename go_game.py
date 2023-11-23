import numpy as np

args = [9,9]

class Piece:
    def __init__(self, action, player, state, args) -> None:
        self.args = args
        self.state = state
        self.action = action # (x, y)
        self.player = player # 1 or -1
        self.x_dim = args[0]
        self.y_dim = args[1]
        self.neighbors = self.get_neighbors(action) # list of neighbour coordinates
        self.group = self.search_group(action, player, state)


    def get_neighbors(self, action):
        x = action[0]
        y = action[1]
        neighbors = [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]
        neighbors = [neighbor for neighbor in neighbors if neighbor[0] >= 0 and neighbor[0] < self.x_dim and neighbor[1] >= 0 and neighbor[1] < self.y_dim]
        return neighbors
    

    def search_group(self, action, player, state):

        neighbors = []
        for neighbor in self.neighbors:
            if state[neighbor[0]][neighbor[1]] != 0 and state[neighbor[0]][neighbor[1]].player == player:
                neighbors.append(state[neighbor[0]][neighbor[1]])
        
        # if neighbors has no pieces of same player
        if neighbors == []:
            group =Group(action, player, state, args)
            group.add_piece(self)
            return group

        groups = []

        for neighbor in neighbors:
            if neighbor != 0 and neighbor.player == player: # if neighbor is same player and not empty
                groups.append(neighbor.group)

        if groups == []:
            group = Group(action, player, state, args)
            group.add_piece(self)
            return group
        
        else:
            if len(groups) > 1:
                for group in groups[1:]:
                    groups[0].merge_group(group)
        
        
        groups[0].add_piece(self)
        return groups[0]
    
    def __str__(self) -> str:
        return str(self.player)
        

class Group:
    def __init__(self, action, player, state, args) -> None:
        self.args = args
        self.state = state
        self.action = action
        self.player = player
        self.x_dim = args[0]
        self.y_dim = args[1]
        self.pieces = []
        self.liberties = []

    def add_piece(self, piece):
        self.pieces.append(piece)
        piece.group = self
        self.liberties = self.search_liberties(self.player, self.state)

    def search_liberties(self, player, state):
        # print("SEARCHING LIBERTIES\n")
        # print(self.pieces)
        liberties = []
        for piece in self.pieces:
            # print(piece)
            # print(piece.neighbors)
            for neighbor in piece.neighbors:
                print(neighbor)
                if state[neighbor[0]][neighbor[1]] == 0:
                    print("LIBERTY FOUND at " + str(neighbor)) # ERROR HERE SOME LIBERTIES ARE NOT ADDED
                    liberties.append(neighbor)
        print("LIBERTIES: " + str(liberties))
        return liberties
    
    def merge_group(self, group):
        self.pieces += group.pieces

        for piece in group.pieces:
            piece.group = self

        return self


class Go:
    def __init__(self, args) -> None:
        self.args = args
        self.x_dim = args[0]
        self.y_dim = args[1]
        self.state = self.get_initial_state()
        self.player = 1


    def get_initial_state(self):
        board = []
        for i in range(self.x_dim):
            board.append([])
            for j in range(self.y_dim):
                board[i].append(0)
        return board
    
    def get_next_state(self, state, action, player):
        next_state = state.copy()
        print("ACTION: " + str(action))
        piece = Piece(action, player, state, self.args)
        print("NEIGHBOURS: " + str(piece.neighbors))
        print("GROUP PIECES: " + str(piece.group.pieces))
        print("GROUP LIBERTIES: " + str(piece.group.liberties))
        print("PLAYER: " + str(piece.player))
        next_state[action[0]][action[1]] = piece
        return next_state

    def print_board(self, state):
        for i in range(len(state)):
            for j in range(len(state[1])):
                print(str(state[i][j]), end=" ")
            print()

    def change_player(self, player):
        return -player
    
    def get_valid_actions(self, state, player):
        valid_actions = []
        for i in range(len(state)):
            for j in range(len(state[1])):
                if state[i,j] == 0:
                    valid_actions.append((i,j))
        return valid_actions
    
    def get_winner(self, state):
        pass



go = Go(args)
state = go.get_initial_state()


while True:
    action = input("Input move (x,y): \n")

    action = action.split(",")
    action = (int(action[0]), int(action[1]))
    state = go.get_next_state(state, action, go.player)
    go.print_board(state)

    go.player = go.change_player(go.player)