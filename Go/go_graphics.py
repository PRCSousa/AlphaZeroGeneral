import pygame

data={'player1':(201,153,255),
      'player2':(179,236,255),
      }

SIZE_BOARD = 9
BLACK = (0,0,0)
WHITE = (255,255,255)
GREEN = (140, 217, 166)


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
    pygame.draw.rect(screen, GREEN, rect=(SCREEN_PADDING, SCREEN_PADDING, CELL_SIZE*(SIZE_BOARD-1), CELL_SIZE*(SIZE_BOARD-1)))
    for i in range(SIZE_BOARD):
        pygame.draw.line(screen, BLACK,(to_pixels(i),SCREEN_PADDING),(to_pixels(i),CELL_SIZE*(SIZE_BOARD-1) + SCREEN_PADDING),3)
        pygame.draw.line(screen, BLACK,(SCREEN_PADDING,to_pixels(i)),(CELL_SIZE*(SIZE_BOARD-1)+SCREEN_PADDING,to_pixels(i)),3)

def draw_piece(x,y,player):
    color = BLACK if player == -1 else WHITE
    pygame.draw.circle(screen,color,(to_pixels(x),to_pixels(y)),PIECE_SIZE)
    pygame.draw.circle(screen,BLACK,(to_pixels(x),to_pixels(y)),PIECE_SIZE,3)

def hover_to_select(player,valid_moves,click):
    mouse_x, mouse_y = pygame.mouse.get_pos()
    x, y = None, None
    if ([to_coord(mouse_x), to_coord(mouse_y)] in valid_moves):
        x, y = to_coord(mouse_x), to_coord(mouse_y)
    
    if (x!=None):
        pixels = (to_pixels(x),to_pixels(y))
        distance = pygame.math.Vector2(pixels[0] - mouse_x, pixels[1] - mouse_y).length()
        if distance < PIECE_SIZE:
            s = pygame.Surface((SCREEN_SIZE, SCREEN_SIZE), pygame.SRCALPHA)
            if player == 1:
                pygame.draw.circle(s,(255,255,255,200),(to_pixels(x),to_pixels(y)),PIECE_SIZE)
            if player == -1:
                pygame.draw.circle(s,(0,0,0,200),(to_pixels(x),to_pixels(y)),PIECE_SIZE)
            pygame.draw.circle(s,BLACK,(to_pixels(x),to_pixels(y)),PIECE_SIZE,3)
            screen.blit(s, (0, 0))
        if click:
            cur_pieces.append([x, y, player])
            valid_moves.remove([x, y])
            return [x, y, -1*player] # might be an issue here
    return [None, None, player]

click = False
valid_moves = []
for i in range(SIZE_BOARD):
    for j in range(SIZE_BOARD):
        valid_moves.append([i, j])

cur_pieces = []
player = 1

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
        if event.type == pygame.MOUSEBUTTONDOWN:
            click = True
        if event.type == pygame.MOUSEBUTTONUP:
            click = False

    screen.fill(GREEN)
    draw_board()

    for piece in cur_pieces:
        draw_piece(piece[0], piece[1], piece[2])

    x, y, player = hover_to_select(player, valid_moves, click)

    pygame.display.flip()

