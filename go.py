import os
import numpy

class Go:

    EMPTY = 0
    BLACK = 1
    WHITE = 2
    MARKER = 4
    LIBERTY = 8

    liberties = []
    block = []
    seki_count = 0
    seki_liberties = []
    run = True
    board = None

    white_captures = 0
    black_captures = 0

    def __init__(self, args) -> None:
        self.board_size = int(args[0])
        self.komi = float(args[1])
        self.board = self.get_initial_state()
        self.color = self.BLACK
        self.move_count = 0
        self.game_over = False
        self.winner = None
        self.history = []

    def get_initial_state(self):
        board = []
        for i in range(self.board_size):
            board.append([])
            for j in range(self.board_size):
                board[i].append(0)
        return board


    def count(self,x,y,colour):
        
        #initialize piece
        piece = self.board[y][x]
        #if there's a stone at square
        if piece and piece & colour and (piece & self.MARKER) == 0:
            #save stone coords
            self.block.append((y,x))
            #mark the stone
            self.board[y][x] |= self.MARKER
            
            #look for neighbours recursively
            if y-1 >= 0: self.count(x,y-1,colour) #walk north
            if x+1 < len(self.board): self.count(x+1,y,colour) #walk east
            if y+1 < len(self.board): self.count(x,y+1,colour) #walk south
            if x-1 >= 0: self.count(x-1,y,colour) #walk west

        #if square is empty
        elif piece == self.EMPTY:
            #mark liberty
            self.board[y][x] = self.LIBERTY
            #save liberties
            self.liberties.append((y,x))

        # print("Liberties: " + str(len(self.liberties)) + " in: " + str(x) + "," + str(y))
        # print("Block: " + str(len(self.block)) + " in: " + str(x) + "," + str(y))
        return self.liberties    

    #remove captured stones
    def clear_block(self):

        #clears the elements in the block of elements which is captured
        for i in range(len(self.block)): 
            y = self.block[i][0]
            x = self.block[i][1]
            self.board[y][x] = self.EMPTY

    #clear groups
    def clear_groups(self):

        #clear block and liberties lisits
        self.block = []
        self.liberties = []

    #restore board after counting stones and liberties
    def restore_board(self):
        #clear groups
        self.clear_groups()

        #unmark stones
        for y in range(len(self.board)):
            for x in range(len(self.board)):
                #restore piece
                val = self.board[y][x]
                new_val = val & 3
                self.board[y][x] = new_val

    #clear board
    def clear_board(self):
        #clear groups
        self.clear_groups()

        #clears the board
        for y in range(len(self.board)):
            for x in range(len(self.board)):
                self.board[y][x] = 0

    def draw_board(self) -> None:

        # Print column coordinates
            print("   ", end="")
            for j in range(len(self.board[0])):
                print(f"{j:2}", end=" ")
            print("\n  +", end="")
            for _ in range(len(self.board[0])):
                print("---", end="")
            print()

            # Print rows with row coordinates
            for i in range(len(self.board)):
                print(f"{i:2}|", end=" ")
                for j in range(len(self.board[0])):
                    print(f"{str(self.board[i][j]):2}", end=" ")
                print()

    def captures(self,color):
        check = False

        #loop over the board squares
        for y in range(len(self.board)):
            for x in range(len(self.board)):
                
                #init piece
                piece = self.board[y][x]

                #if stone belongs to given colour
                if piece & color:
                    
                    #count liberties
                    self.count(x,y,color)

                    #if no liberties remove the stones
                    if len(self.liberties) == 0: 
                        
                        prisioners = len(self.block)
                        #clear block
                        self.clear_block()

                        if color == self.BLACK:
                            self.black_captures += prisioners
                        else:
                            self.white_captures += prisioners


                        #if the move is a "ko" move but causes the capture of stones, then it is not allowed, unless it is the second move, in which case it is dealt afterwards
                        if self.seki_count == 0:
                            print("Seki Found")
                            #returns False, which means that the move has caused a capture (the logic worked out that way in the initial development and i'm not sure what it would affect if it is changed)
                            check = True
                            self.seki_count = 1
                            continue

                    #restore the board
                    self.restore_board()
        # print("Seki Count: " + str(self.seki_count))
        #print("Captures: " + str(check))
        return check

    def change_player(self):
        
            #switching colours for the next move
            if self.color == self.BLACK:
                self.color = self.WHITE
            else:
                self.color = self.BLACK

    def get_player(self):
        return self.color

    def set_stone(self,y,x):
        #making move on the board
        self.board[y][x] = self.color

    def is_valid_move(self, action):

        pre_board = numpy.copy(self.board)
        a, b = action
        self.set_stone(a,b)

        self.captures(3 - self.get_player())
        str_board = self.board_to_str()

        if str_board in self.history:
            print("Invalid Move: Repeated State")
            self.board = numpy.copy(pre_board)
            return False

        if self.seki_count == 1:
            if str_board in self.history:
                print("Invalid Move: Repeated State")
                self.board = numpy.copy(pre_board)
                return False
        else:
            if self.captures(3-self.get_player()) == False and self.captures(self.get_player()) == True:
                print("Invalid Move: Suicide")
                self.board = numpy.copy(pre_board)
                return False
            
        self.board = numpy.copy(pre_board)  
        return True
    
    def board_to_str(self):
        return ''.join([''.join(map(str, row)) for row in self.board])

    def save_state(self):
        str_board = self.board_to_str()
        self.history.append(str_board)  # Save string to history list

    def get_next_state(self, action):
        a, b = action
        # checking if the move is part of is the secondary move to a ko fight
        self.set_stone(a, b)
        self.captures(3-self.get_player())
        self.save_state()
        return self.board

    def check_board_full(self):

        empty_count = 0
        for row in self.board:
            for stone in row:
                if stone == 0:
                    empty_count += 1
                if empty_count == 2:
                    return True
                
        return False

    def check_win_and_over(self):

        if self.check_board_full():
            if self.black_captures >= self.white_captures + self.komi:
                return 1, True
            else: 
                return 1, False
        
        return 1, False
    
    def check_score(self):
        black_pieces, white_pieces = 0
        for row in self.board:
            for stone in row:
                if stone == 1:
                    black_pieces += 1
                if stone == 2:
                    white_pieces += 1
        
        black_points = black_pieces + self.black_captures
        white_points = white_pieces + self.white_captures + self.komi

        return black_points - white_points
    
    def get_value_and_terminated(self):
        return self.check_score(), self.check_board_full()
    
    def get_valid_moves(self):
        possible_moves = set()

        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.board[i][j] == 0:
                    possible_moves.add((i, j))
        
        return possible_moves


args = [9, 5.5]

go = Go(args)

go.draw_board()

while True:
    print("Player: " + str(go.get_player()))
    print("Input: ")
    a, b = tuple(int(x.strip()) for x in input().split(' '))
    action = (a, b)
    if go.is_valid_move(action):
        go.board = go.get_next_state(action)
        go.change_player()
        

    go.draw_board()
