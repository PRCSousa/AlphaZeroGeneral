import numpy as np
from copy import deepcopy


class Piece:
    def __init__(self, action, player, state, args) -> None:
        self.args = args
        self.state = state
        self.action = action  # (x, y)
        self.player = player  # 1 or -1 or 0 (for territory calculations)
        self.x_dim = args[0]
        self.y_dim = args[1]
        self.neighbors = self.get_neighbors(
            action)  # list of neighbour coordinates
        self.group = self.search_group(action, player, state)

    def get_neighbors(self, action):
        x = action[0]
        y = action[1]
        neighbors = [(x-1, y), (x+1, y), (x, y-1), (x, y+1)]
        neighbors = [neighbor for neighbor in neighbors if neighbor[0] >=
                     0 and neighbor[0] < self.x_dim and neighbor[1] >= 0 and neighbor[1] < self.y_dim]
        return neighbors

    def search_group(self, action, player, state):

        neighbors = []
        for neighbor in self.neighbors:
            if state[neighbor[0]][neighbor[1]] != 0 and state[neighbor[0]][neighbor[1]].player == player:
                neighbors.append(state[neighbor[0]][neighbor[1]])

        # if neighbors has no pieces of same player
        if neighbors == []:
            group = Group(action, player, state, args)
            group.add_piece(self)
            return group

        groups = []

        for neighbor in neighbors:
            if neighbor != 0 and neighbor.player == player:  # if neighbor is same player and not empty
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

    def search_liberties(self, player, state):
        #print("SEARCHING LIBERTIES\n")
        # print(self.pieces)
        liberties = []
        for piece in self.pieces:
            # print(piece)
            # print(piece.neighbors)
            for neighbor in piece.neighbors:

                # print("CHECKING " +str(neighbor))
                # print(state[neighbor[0]][neighbor[1]])
                if state[neighbor[0]][neighbor[1]] == 0:

                    # print("LIBERTY FOUND at " + str(neighbor))
                    liberties.append(neighbor)
        # print("LIBERTIES: " + str(liberties))
        return set(liberties)

    def merge_group(self, group):
        self.pieces += group.pieces

        for piece in group.pieces:
            piece.group = self

        return self

    def capture(self, state):
        quant = len(self.pieces)
        for piece in self.pieces:
            state[piece.action[0]][piece.action[1]] = 0
            piece.group = None
        return quant


# ##################################################################### #

class Go:
    def __init__(self, args) -> None:
        self.args = args
        self.x_dim = args[0]
        self.y_dim = args[1]
        self.komi = args[2]
        self.state = self.get_initial_state()
        self.player = 1
        self.previous_equals = False
        self.statelist = []
        self.prisioners = [0,0]

    def get_initial_state(self):
        board = []
        for i in range(self.x_dim):
            board.append([])
            for j in range(self.y_dim):
                board[i].append(0)
        return board

    def put_piece(self, state, action, piece: Piece):

        pri = 0

        state[action[0]][action[1]] = piece  # temporary for checking
        piece.group.liberties = piece.group.search_liberties(
            piece.player, state)  # update liberties4,4
        # if it doesn't capture antything, remove piece and print suicide
        for neighbor in piece.neighbors:
            if state[neighbor[0]][neighbor[1]] != 0:
                state[neighbor[0]][neighbor[1]].group.liberties = state[neighbor[0]
                                                                        ][neighbor[1]].group.search_liberties(state[neighbor[0]][neighbor[1]], state)
                if state[neighbor[0]][neighbor[1]].player != piece.player:
                    if len(state[neighbor[0]][neighbor[1]].group.liberties) == 0:
                        pri += state[neighbor[0]][neighbor[1]].group.capture(state)
        
        self.prisioners[0 if piece.player == -1 else 1] += pri

        return state

    def suicide(self, state, piece):

        # deepcopy the board to verify suicide
        copystate = deepcopy(state)
        copypiece = deepcopy(piece)

        copystate[action[0]][action[1]] = copypiece  # temporary
        copypiece.group.liberties = copypiece.group.search_liberties(
            copypiece.player, copystate)
        # if it has more than 0 liberties, no suicide
        if (len(copypiece.group.liberties) > 0):
            return False

        # if it removes a enemy group, liberties above 0 so legal
        for neighbor in copypiece.neighbors:
            if copystate[neighbor[0]][neighbor[1]] != 0:
                copystate[neighbor[0]][neighbor[1]].group.liberties = copystate[neighbor[0]][neighbor[1]
                                                                                             ].group.search_liberties(copystate[neighbor[0]][neighbor[1]].player, copystate)
                if len(copystate[neighbor[0]][neighbor[1]].group.liberties) == 0 and copystate[neighbor[0]][neighbor[1]].player != copypiece.player:
                    # capture group if it has no liberties and is not same player
                    return False

        return True

    def get_next_state(self, state, action, player):
        next_state = state.copy()

        if go.check_skip(state, action, go.player):
            if self.previous_equals:
                self.get_winner(state)
                return -1
            else:
                go.player = go.change_player()
                self.previous_equals = True
                return state

        # print("ACTION: " + str(action))
        piece = Piece(action, player, state, self.args)
        if self.suicide(state, piece):
            print("Suicide is an illegal move")
            return state

        # print("NEIGHBOURS: " + str(piece.neighbors))
        # print("PLAYER: " + str(piece.player))
        next_state = self.put_piece(next_state, action, piece)
        statemat = self.convert_state_to_matrix(next_state)
        #print("ADDING TO REPEAT CHECK: ")
        self.add_matrix_to_positions(statemat)
        go.player = go.change_player()
        self.previous_equals = False

        # print("GROUP PIECES: " + str(piece.group.pieces))
        # print("GROUP LIBERTIES: " + str(piece.group.liberties))

        return next_state

    def is_valid_move(self, state, action, player):
        x, y = action

        statecopy = deepcopy(state)
        temppiece = Piece(action, player, statecopy, args)

        if state[x][y] != 0:
            return False

        self.put_piece(statecopy, action, temppiece)

        # print("NEW TEMPORARY STATE:")
        statecopy = self.convert_state_to_matrix(statecopy)
        # print(str(statecopy))
        # print("VERIFYING REPEATED LIST: ")
        # print(str(self.statelist))
        if any(np.array_equal(statecopy, stateelement) for stateelement in self.statelist):
            print("Invalid Move: Repeated State")
            return False
        return True

    def print_board(self, state):
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

    def change_player(self):
        return -self.player

    def get_valid_actions(self, state, player):
        valid_actions = []
        for i in range(len(state)):
            for j in range(len(state[1])):
                if state[i, j] == 0:
                    valid_actions.append((i, j))
        return valid_actions

    def check_skip(self, state, action, player):
        if (action == (-1, -1)):
            print("Player " + str(player) + " skips")
            return True

    def evaluate(self, state, player):
        pieces = 0
        territory = 0
        prisioners = self.prisioners[0 if player == -1 else 1]
        territory_spaces = set()

        for i in range(self.x_dim):
            for j in range(self.y_dim):
                if state[i][j] != 0 and state[i][j].player == player:
                    pieces += 1
                    for neighbor in state[i][j].neighbors:
                        piece = Piece((neighbor[0],neighbor[1]), 0, state, args)
                        state[neighbor[0]][neighbor[1]] = piece
                        sum = 0
                        for adj in piece.neighbors:
                            if state[adj[0]][adj[1]] != 0:
                                sum += 1
                        if sum > 0 and abs(sum) >= 2:
                            territory_spaces.add((neighbor[0],neighbor[1]))

                    territory = len(territory_spaces)

        return pieces, territory, prisioners

    def get_winner(self, state):
        p1, t1, pri1 = self.evaluate(state, 1)
        p2, t2, pri2 = self.evaluate(state, -1)

        if p1 > p2:
            print("Player 1 wins with " + str(p1) + " stones, " +  str(pri1) + " prisioners and " + str(t1) + " of territory against " + str(p2) + " stones, " + str(pri2) + " prisioners and " + str(t2) + " of territory")
        elif p2 > p1:
            print("Player -1 wins with " + str(p2) + " stones, " + str(pri2) + " prisioners and " + str(t2) + " of territory against " + str(p1) + " stones, " + str(pri1) + " prisioners and " + str(t1) + " of territory")
    def add_matrix_to_positions(self, matrix):
        # print(str(matrix))
        self.statelist.append(matrix)

    def convert_state_to_matrix(self, state):
        mat = np.zeros((self.x_dim, self.y_dim))
        for i in range(len(state)):
            for j in range(len(state[1])):
                if str(state[i][j]) == "1":
                    mat[i][j] = 1
                elif str(state[i][j]) == "-1":
                    mat[i][j] = -1
        return mat


args = [5, 5, 5.5] # x, y, komi
go = Go(args)
state = go.get_initial_state()

while True:
    go.print_board(state)

    action = input("Input move x,y | -1,-1 to pass: \n")
    action = action.split(",")

    try:
        action = (int(action[0]), int(action[1]))
        while (action[0] >= go.x_dim or action[1] >= go.y_dim or action[0] < -1 or action[1] < -1):
            print("Invalid Move: Out of Bounds")
            action = input("Input move x,y | -1,-1 to pass: \n")
            action = action.split(",")
            action = (int(action[0]), int(action[1]))

    except:
        print("Invalid Move: Not a number")
        continue

    while (state[action[0]][action[1]] != 0 or go.is_valid_move(state, action, go.player) == False):
        action = input("Input move x,y | -1,-1 to pass: \n")
        action = action.split(",")
        action = (int(action[0]), int(action[1]))

    state = go.get_next_state(state, action, go.player)

    if state == -1:
        break
