import pygame

data={'size': 9,
      'player1':(201,153,255),
      'player2':(179,236,255),
      'white':(255,255,255),
      'black':(0,0,0),
      'green_go':(140, 217, 166)
      }

pygame.init()
pygame_icon = pygame.image.load('image.png')
pygame.display.set_icon(pygame_icon)

SIDE=800

screen=pygame.display.set_mode((SIDE,SIDE))

pygame.display.set_caption("depression")

def draw_board(num_size):
    for i in range(data['size']):
        pygame.draw.line(screen, data['black'],(SIDE*i//data['size'],0),(SIDE*i//data['size'],SIDE),3)
        pygame.draw.line(screen, data['black'],(0,SIDE*i//data['size']),(SIDE,SIDE*i//data['size']),3)
def draw_piece(x,y,player):
    color=data['white']
    if player==-1:
        color=data['black']
    pygame.draw.circle(screen,color,(x*SIDE//data['size'],y*SIDE//data['size']),30)
    pygame.draw.circle(screen,data['black'],(x*SIDE//data['size'],y*SIDE//data['size']),30,3)

def hover_piece(x, y, player):
    mouse_x, mouse_y = pygame.mouse.get_pos()
    coords = (x*SIDE//data['size'],y*SIDE//data['size'])
    distance = pygame.math.Vector2(coords[0] - mouse_x, coords[1] - mouse_y).length()
    if distance < 30:
        s = pygame.Surface((SIDE, SIDE), pygame.SRCALPHA)
        if player == 1:
            pygame.draw.circle(s,(0,0,0,30),(x*SIDE//data['size'],y*SIDE//data['size']),30)
        if player == -1:
            pygame.draw.circle(s,(255,255,255,60),(x*SIDE//data['size'],y*SIDE//data['size']),30)
        screen.blit(s, (0, 0))

    pygame.draw.circle(screen,data['black'],(x*SIDE//data['size'],y*SIDE//data['size']),30,3)

def hover_to_select(x,y,player):
    mouse_x, mouse_y = pygame.mouse.get_pos()
    coords = (x*SIDE//data['size'],y*SIDE//data['size'])
    distance = pygame.math.Vector2(coords[0] - mouse_x, coords[1] - mouse_y).length()
    if distance < 30:
        s = pygame.Surface((SIDE, SIDE), pygame.SRCALPHA)
        if player == 1:
            pygame.draw.circle(s,(0,0,0,255),(x*SIDE//data['size'],y*SIDE//data['size']),30)
        if player == -1:
            pygame.draw.circle(s,(255,255,255,255),(x*SIDE//data['size'],y*SIDE//data['size']),30)
        pygame.draw.circle(s,data['black'],(x*SIDE//data['size'],y*SIDE//data['size']),30,3)
        screen.blit(s, (0, 0))

    

def select_piece(valid_moves, player):
    for move in valid_moves:
        mouse_x, mouse_y = pygame.mouse.get_pos()
        coords = (move[0]*SIDE//data['size'],move[1]*SIDE//data['size'])
        distance = pygame.math.Vector2(coords[0] - mouse_x, coords[1] - mouse_y).length()
        if distance < 30:
            if player == 1:
                pygame.draw.circle(screen,(0,0,0,30),(coords[0],coords[1]),30)
            if player == -1:
                pygame.draw.circle(screen,(255,255,255),(coords[0],coords[1]),30)
            return
    print('invalid')
    ...

while True:
    valid_moves = [[2, 1],[7, 4], [3,3], [6,4]]
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()


    screen.fill(data['green_go'])

    for i in range(data['size']):
        pygame.draw.line(screen, data['black'],(SIDE*i//data['size'],0),(SIDE*i//data['size'],SIDE),3)
        pygame.draw.line(screen, data['black'],(0,SIDE*i//data['size']),(SIDE,SIDE*i//data['size']),3)
    draw_piece(1,1,-1)
    draw_piece(2,4,1)
    hover_piece(1,1,-1)
    hover_piece(2,4,1)

    for move in valid_moves:
        hover_to_select(move[0], move[1], 1)

    left, middle, right = pygame.mouse.get_pressed()
    if left:
        select_piece(valid_moves, 1)

    pygame.display.flip()

