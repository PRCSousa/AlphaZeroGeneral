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

while True:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()

    screen.fill(data['white'])

    for i in range(data['size']):
        pygame.draw.line(screen, data['black'],(SIDE*i//data['size'],0),(SIDE*i//data['size'],SIDE),3)
        pygame.draw.line(screen, data['black'],(0,SIDE*i//data['size']),(SIDE,SIDE*i//data['size']),3)


    font = pygame.font.Font(None, 80)
    img = font.render("ok", True, data['black'])
    text_rect = img.get_rect(center=(SIDE//2, SIDE//2))
    screen.blit(img, text_rect)

    pygame.display.flip()

