import numpy as np
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
                    if empty >= self.column_count * self.row_count // 4: # if more than 1/4 of the board is empty, it is not the endgame
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
                    if empty >= self.column_count * self.row_count // 6:
                        endgame = False
                        break

        black, white = self.count_influenced_territory_enhanced(state)
                            
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