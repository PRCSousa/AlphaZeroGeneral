import pygame

data={'player1':(201,153,255),
      'player2':(179,236,255),
      }

SIZE_BOARD = 9
RED = (238, 167, 255)
BLUE = (113, 175, 255)
GRAY = (115, 115, 115)
BLACK = (0, 0, 0)

pygame.init()
pygame_icon = pygame.image.load('image.png')
pygame.display.set_icon(pygame_icon)

SCREEN_SIZE=600
SCREEN_PADDING = 50
CELL_SIZE = (SCREEN_SIZE - SCREEN_PADDING) // SIZE_BOARD
PIECE_SIZE = (SCREEN_SIZE - 2*SCREEN_PADDING) // SIZE_BOARD // 3

screen=pygame.display.set_mode((SCREEN_SIZE,SCREEN_SIZE))

pygame.display.set_caption("Go")

def to_pixels(x):
    return SCREEN_PADDING + x*CELL_SIZE

def to_coord(x):
    quarter = CELL_SIZE//4
    closest = (x-SCREEN_PADDING)//CELL_SIZE
    if abs(to_pixels(closest)-(x-SCREEN_PADDING > to_pixels(closest)-(x-SCREEN_PADDING+quarter))):
        closest = (x-SCREEN_PADDING+quarter)//CELL_SIZE
    return closest

def draw_board():
    pygame.draw.rect(screen, GRAY, rect=(SCREEN_PADDING, SCREEN_PADDING, CELL_SIZE*(SIZE_BOARD-1), CELL_SIZE*(SIZE_BOARD-1)))
    for i in range(SIZE_BOARD + 1):

        pygame.draw.line(screen, BLACK,
                         (to_pixels(i) - CELL_SIZE / 2, SCREEN_PADDING // 2 + - 5),
                         (to_pixels(i) - CELL_SIZE / 2, CELL_SIZE*(SIZE_BOARD) + SCREEN_PADDING - CELL_SIZE / 2),
                         3)


        pygame.draw.line(screen, BLACK,
                         (SCREEN_PADDING // 2 - 7, to_pixels(i) - CELL_SIZE / 2),
                         (CELL_SIZE*(SIZE_BOARD) + SCREEN_PADDING - CELL_SIZE / 2, to_pixels(i) - CELL_SIZE / 2),
                         3)


def draw_piece(x,y,player):
    color = RED if player == -1 else BLUE
    pygame.draw.circle(screen,color,(to_pixels(x),to_pixels(y)),PIECE_SIZE)
    pygame.draw.circle(screen,BLACK,(to_pixels(x),to_pixels(y)),PIECE_SIZE,3)


def hover_to_select(sel_x, sel_y, player, valid_moves, click, selected_piece, cur_pieces, last_click_time):

    current_time = pygame.time.get_ticks()
    mouse_x, mouse_y = pygame.mouse.get_pos()
    x, y = None, None
    if ([to_coord(mouse_x), to_coord(mouse_y), player] in cur_pieces):
        x, y = to_coord(mouse_x), to_coord(mouse_y)

        if click and current_time - last_click_time > 100:  # 100 milliseconds debounce
            if selected_piece:
                if x == sel_x and y == sel_y: #deselection
                    selected_piece = False
                    sel_x = -1
                    sel_y = -1
                    print("cancel")
                    
            else: #selection
                print("select")
                selected_piece = True
                sel_x = x
                sel_y = y

            last_click_time = current_time
    
    if click and current_time - last_click_time > 100 and [to_coord(mouse_x), to_coord(mouse_y)] in valid_moves and selected_piece:
        print("thing happen")
        cur_pieces.append([to_coord(mouse_x), to_coord(mouse_y),player])
        player = -player
        selected_piece = False
    
    # Draw hollow circles on valid moves if a piece is selected
    if selected_piece:
        for move in valid_moves:
            px, py = to_pixels(move[0]), to_pixels(move[1])
            s = pygame.Surface((SCREEN_SIZE, SCREEN_SIZE), pygame.SRCALPHA)
            pygame.draw.circle(s, (113, 175, 255, 100) if player == 1 else (238, 167, 255, 100), (px, py), PIECE_SIZE, 3)  # Change color as needed
            screen.blit(s, (0, 0))


    return [sel_x, sel_y, player, selected_piece, last_click_time]

# Call the function with an additional 'selected_piece' parameter, initially None
# selected_piece = hover_to_select(player, valid_moves, click, selected_piece, cur_pieces)


click = False
valid_moves = []
for i in range(SIZE_BOARD):
    for j in range(SIZE_BOARD):
        valid_moves.append([i, j])

cur_pieces = [[0,0,1], [8,8,-1]]
player = -1
selected_piece = False
last_click_time = 0
x, y = -1, -1
while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
        if event.type == pygame.MOUSEBUTTONDOWN:
            click = True
        if event.type == pygame.MOUSEBUTTONUP:
            click = False

    screen.fill(GRAY)
    draw_board()

    for piece in cur_pieces:
        draw_piece(piece[0], piece[1], piece[2])

    x, y, player, selected_piece, last_click_time = hover_to_select(x, y, player, valid_moves, click, selected_piece, cur_pieces, last_click_time)
    if x != -1 and y != -1:
        # aqui vou buscar a lista de valid moves para [x,y]
        # valid_moves = []
        pass 
    pygame.display.flip()