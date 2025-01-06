import pygame
import math
import os
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)

pygame.init()

sf=6 #scale factor

size = (28*sf, 28*sf)
screen = pygame.display.set_mode(size)

canvas = []
for i in range(28):
    l = []
    for j in range(28):
        l.append(0)
    canvas.append(l)

def clear_canvas():
    for i in range(28):
        for j in range(28):
            canvas[i][j]=0
            
print(canvas)

def predict():
    x = torch.tensor(canvas)
    model.eval()
    with torch.no_grad():
        output = model(test_data[img_num][0].view(1,1,28,28))

    return(output.argmax().item())

pygame.display.set_caption("Drawing!")

# Used to manage how fast the screen updates
clock = pygame.time.Clock()

done=False
 
# -------- Main Program Loop -----------
while not done:
    # --- Main event loop
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            done = True

    #mouse and keyboard updates
    if(pygame.mouse.get_pressed()[0]==True):
        pos = pygame.mouse.get_pos()
        x=math.floor(pos[0]/sf)
        y=math.floor(pos[1]/sf)
        canvas[y][x]=1
        print(x, y)

    keys = pygame.key.get_pressed()
    if keys[pygame.K_SPACE]:
        print('clear')
        clear_canvas()
    if keys[pygame.K_RETURN]:
        print("i gitchue")
        p=predict()
        print(f"and the number is... {p}")

 
    #drawing!
    screen.fill(WHITE)

    for i in range(len(canvas)):
        for j in range(len(canvas[i])):
            if(canvas[i][j]) == 0:
                pygame.draw.rect(screen, WHITE, pygame.Rect(j*sf, i*sf, sf, sf))
            else:
                pygame.draw.rect(screen, BLACK, pygame.Rect(j*sf, i*sf, sf, sf))

 
    # update screen
    pygame.display.flip()
    clock.tick(60)
 
# Close the window and quit.
pygame.quit()
