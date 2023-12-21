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
        self.action_size = self.x_dim * self.y_dim
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
            group = Group(action, player, state, self.args)
            group.add_piece(self)
            return group

        groups = []

        for neighbor in neighbors:
            if neighbor != 0 and neighbor.player == player:  # if neighbor is same player and not empty
                groups.append(neighbor.group)

        if groups == []:
            group = Group(action, player, state, self.args)
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
        self.action_size = self.x_dim * self.y_dim
        self.komi = args[2]
        self.state = self.get_initial_state()
        self.player = 1
        self.previous_equals = False
        self.statelist = []
        self.prisioners = [0, 0]

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
                        pri += state[neighbor[0]][neighbor[1]
                                                  ].group.capture(state)

        self.prisioners[0 if piece.player == -1 else 1] += pri / 2

        return state

    def suicide(self, state, piece, action) -> bool:

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

        if self.check_skip(state, action, self.player):
            if self.previous_equals:
                # print winner
                return -1
            else:
                self.player = self.change_player()
                self.previous_equals = True
                return state

        # print("ACTION: " + str(action))
        piece = Piece(action, player, state, self.args)
        if self.suicide(state, piece, action):
            print("Suicide is an illegal move")
            return state

        # print("NEIGHBOURS: " + str(piece.neighbors))
        # print("PLAYER: " + str(piece.player))
        next_state = self.put_piece(next_state, action, piece)
        statemat = self.convert_state_to_matrix(next_state)
        #print("ADDING TO REPEAT CHECK: ")
        self.add_matrix_to_positions(statemat)
        self.player = self.change_player()
        self.previous_equals = False

        # print("GROUP PIECES: " + str(piece.group.pieces))
        # print("GROUP LIBERTIES: " + str(piece.group.liberties))

        return next_state

    def is_valid_move(self, state, action, player):
        x, y = action

        statecopy = deepcopy(state)
        temppiece = Piece(action, player, statecopy, self.args)

        if state[x][y] != 0:
            return False

        self.put_piece(statecopy, action, temppiece)

        if self.suicide(statecopy, temppiece, action):
            return False

        # print("NEW TEMPORARY STATE:")
        statecopy = self.convert_state_to_matrix(statecopy)
        # print(str(statecopy))
        # print("VERIFYING REPEATED LIST: ")
        # print(str(self.statelist))
        if any(np.array_equal(statecopy, stateelement) for stateelement in self.statelist):
            #print("Invalid Move: Repeated State")
            return False
        return True

    def print_board(self, state):
        print("\nEvaluation: 1 | %.1f | - | %.1f | -1" %(self.get_score( state, 1), self.get_score(state, -1)))
        print("Prisioners: 1 | %.1f | - | %.1f | -1\n" %(self.prisioners[1], self.prisioners[0]))
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

    def get_valid_moves(self, state, player):
        valid_actions = []
        for i in range(len(state)):
            for j in range(len(state[1])):
                if state[i][j] == 0 and self.is_valid_move(state, (i, j), player):
                    valid_actions.append((i, j))
        return valid_actions
    
    def check_available_moves(self, state, player):
        if len(self.get_valid_moves(state, player)) > 0:
            return True
        else:
            return False


    def check_skip(self, state, action, player):
        if (action == (-1, -1)):
            print("Player " + str(player) + " skips")
            return True

    def evaluate(self, state, player):

        statecopy = deepcopy(state)

        pieces = 0
        prisioners = self.prisioners[0 if player == -1 else 1]

        for i in range(self.x_dim):
            for j in range(self.y_dim):
                pieces += 1 if statecopy[i][j] == player else 0

        return pieces, prisioners
                
    
    def get_score(self, state, player):
        p1, t1 = self.evaluate(state, player)
        if player == -1:
            p1 += self.komi

        score = p1 + t1
        return score
    
    def check_win_and_over(self, state):
        if not self.check_available_moves(state, 1) and not self.check_available_moves(state, -1):
            if self.get_score(state, 1) > self.get_score(state, -1):
                return 1, True
            elif self.get_score(state, 1) < self.get_score(state, -1):
                return -1, True
            else:
                return 2, True
        else:
            return 0, False
        
    def get_value_and_terminated(self, state):
        winner, game_over = self.check_win_and_over(state)
        return winner, game_over

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

    def get_encoded_state(self, state):
        encoded_state = np.stack(
            (state == -1, state == 0, state == 1)
        ).astype(np.float32)
        
        return encoded_state