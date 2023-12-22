import numpy as np

class Go():

    EMPTY = 0
    BLACK = 1
    WHITE = -1
    BLACKMARKER = 4
    WHITEMARKER = 5
    LIBERTY = 8

    def __init__(self):
        self.row_count = 9
        self.column_count = 9
        self.komi = 6.5
        self.action_size = self.row_count * self.column_count + 1
        self.liberties = []
        self.block = []
        self.seki_count = 0
        self.seki_liberties = []
        
    def get_initial_state(self):
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

    def print_board(self, state) -> None:
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
                    print(f"{str(state[i][j]):2}", end=" ")
                print()
    
    def captures(self, state: list,player: int) -> tuple[bool, list]:
        '''
        # Description:
        Checks if a move causes a capture of stones of the player passed as an argument.
        If a move causes a capture, the stones are removed from the board.

        # Returns:
        A tuple containing a boolean indicating if a capture has been made and the board with the captured stones removed.
        '''
        check = False

        #loop over the board squares
        for y in range(len(state)):
            for x in range(len(state)):
                
                #init piece
                piece = state[y][x]

                #if stone belongs to given colour
                if piece == player:
                    
                    #count liberties
                    liberties = []
                    block = []
                    liberties, block = self.count(x, y, state, player, liberties, block)

                    #if no liberties remove the stones
                    if len(liberties) == 0: 
                        #clear block

                        state = self.clear_block(block, state)

                        #if the move is a "ko" move but causes the capture of stones, then it is not allowed, unless it is the second move, in which case it is dealt afterwards
                        if self.seki_count == 0:
                            # print("Seki Found")
                            # returns False, which means that the move has caused a capture (the logic worked out that way in the initial development and i'm not sure what it would affect if it is changed)
                            check = True
                            self.seki_count = 1
                            continue

                    #restore the board
                    state = self.restore_board(state)
        # print("Seki Count: " + str(self.seki_count))
        # print("Captures: " + str(check))
        return check, state
    
    def set_stone(self, a, b, state, player):
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
        state = self.captures(state, -player)[1]
        return state
    
    def get_valid_moves(self, state):
        '''
        # Description:
        Returns a matrix with the valid moves for the current player.
        '''
        state = np.array(state)
        state = state.reshape(-1) == 0
        state = np.concatenate([state, [1]])
        return (state).astype(np.uint8)

    def get_value_and_terminated(self, state, action):
        '''
        # Description:
        Returns the value of the state and if the game is over.
        '''
        black_count = np.sum(state == self.BLACK)
        white_count = np.sum(state == self.WHITE)


        if black_count > white_count + self.komi:
            return 1, True
        elif white_count > black_count + self.komi:
            return -1, True
        else:
            return 0, False
        
    def check_win(self, state, action):
        '''
        # Description:
        Checks if the game is over.
        '''
        if self.check_board_full(state):
            return True
        else:
            return False
        
    def check_board_full(self, state: list) -> bool:
        '''
        # Description:
        Checks if the board is full.

        # Returns:
        True if the board is full.
        '''

        empty_count = 0
        for row in state:
            for stone in row:
                if stone == 0:
                    empty_count += 1
                if empty_count >= 3:
                    return False
                
        return True
    
    def check_score(self, state: list) -> float:
        '''
        # Description:
        Checks the score of the game.

        # Returns:
        The score of the game, positive if black is winning, negative if white is winning.
        '''
        black_pieces = 0
        white_pieces = 0
        for row in state:
            for stone in row:
                if stone == 1:
                    black_pieces += 1
                if stone == 2:
                    white_pieces += 1
        
        black_points = black_pieces
        white_points = white_pieces + self.komi

        return black_points - white_points

    def get_opponent(self, player):
        return -player
    
    def get_opponent_value(self, value):
        return -value
    
    def get_encoded_state(self, state):
        layer_1 = np.where(np.array(state) == -1, 1, 0).astype(np.float32)
        layer_2 = np.where(np.array(state) == 0, 1, 0).astype(np.float32)
        layer_3 = np.where(np.array(state) == 1, 1, 0).astype(np.float32)

        result = np.stack([layer_1, layer_2, layer_3]).astype(np.float32)

        return result
    
    def change_perspective(self, state, player):
        return state * player