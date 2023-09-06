import pygame as py
import numpy as np
import time

from model.snake_neural_net import *
from model.agent import *
import torch
from utils.game_state import *
from utils.encoder_decoder import *
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

turn = 0
grey = (192, 192, 192,128)
dark_grey = (48, 48, 48)
dark_brown = (70,40,40)
cream = (230,221,197)
red = (255, 0, 0)
blue = (0,0,255)
black = (0,0,0)
bwidth = 800
bheight = 800
numsqx = 32
numsqy = 32
swidth = bwidth/numsqx
sheight = bheight/numsqy
start_length = 7




def GUI():
    testing = False
    if testing:
        epsilon = 0
        agent = Agent(load=True)
    else:
        epsilon = 1.0
        agent = Agent()
    epsilon_decay = 0.995
    min_epsilon = 0.01

    #agent initialization
    
    buffer = ReplayBuffer(20000)
    moves = 10000
    
    
    #angle_norm = 180
    #dist_norm = np.sqrt(numsqx**2 +numsqy**2)

    #inp = np.zeros(7)

    global turn
    py.init()
    
    gui_board = py.display.set_mode((bwidth, bheight))
    gui_board.fill(black)
    
    
    done = False
    board = Board(numsqx, numsqy)
    snake = Snake(7, (numsqx, numsqy))
    food = Food(numsqx, numsqy)
    board_fill(gui_board,board)
    
    
    
    num_epochs = 2000
    batch_size = 128
    
    for epoch in range(num_epochs):
        direction=snake.move_dir
        eat_self = False
        next_direction = None
        c = 0
        snake.ate_food = False
        snake.dead = False
        state = encode_board(snake, food, board)
        if epoch != 0:
            reset_all(board, gui_board,snake, food)
            done = False
        while not done:
            action = agent.get_action(state, epsilon)
            for event in py.event.get():
                if event.type == py.QUIT:
                    done = True
                elif event.type == py.KEYDOWN:
                    if event.key == py.K_DOWN and direction != 3:
                        direction = 3
                    elif event.key == py.K_UP and direction != 1:
                        direction = 1
                    elif event.key == py.K_LEFT and direction != 0:
                        direction = 0
                    elif event.key == py.K_RIGHT and direction != 2:
                        direction = 2
                    elif event.key == py.K_SPACE:
                        #torch.save(ffnn.state_dict(), 'SnakeGame/model/Saved_Model/snake_model.pt')
                        done = True
                    else:
                        #end_message(gui_board)
                        #time.sleep(3)
                        done = True

            c+=1
            next_state, reward, done = step(snake, board, gui_board, food, action)
            
            if c==moves:
                #reset game
                done = True

            if not testing:
                buffer.push(state, action,reward, next_state, done)
                #model train
                if len(buffer) > batch_size:
                    agent.train(buffer, batch_size)
            else:
                pass
        if not testing:
            epsilon = max(min_epsilon, epsilon*epsilon_decay)
            

            

    model = agent.return_model()
    save_model(model)
    py.quit()


def step(snake:Snake, board:Board, gui_board, food:Food, action):
    snake.dead = False
    old_snake = copy.deepcopy(snake)
    
    end,eat_self = update_all(snake,board,gui_board,food,action)
    snake.new_dir(action)
    next_state = encode_board(snake,food, board)
    reward = get_reward(old_snake,food,snake)
    done = False
    if end or eat_self:
        #reset game
        done = True
    #time.sleep(0.1)
    return next_state, reward, done


def board_fill(gui_board, board: Board):
    py.display.update()
    for i in range(0, board.x):
        for j in range(0,board.y):
            py.draw.rect(gui_board, black, (i*swidth, j*sheight, swidth, sheight))
            py.draw.rect(gui_board, dark_grey, (i*swidth+1, j*sheight+1, swidth-1, sheight-1))
    py.display.update()


def reset_all(board:Board, gui_board, snake:Snake, food:Food):
    board.reset()
    snake.reset()
    food.reset()
    board_fill(gui_board,board)
    draw_snake(gui_board,snake)
    draw_food(gui_board, food)



def draw_snake(gui_board,snake:Snake):
    for s in snake.points:
        py.draw.rect(gui_board, red, (s[0]*swidth+1, s[1]*sheight+1, swidth-1, sheight-1))
    py.display.update()
    

def update_snake(gui_board,board, direction, snake:Snake):
    board_fill(gui_board,board)
    snake.update(move_dir=direction)
    
    draw_snake(gui_board, snake)

def grow_snake(gui_board,board, snake:Snake):
    board_fill(gui_board,board)
    snake.grow()

def update_all(snake:Snake, board:Board, gui_board,food:Food, direction):
    check_eaten(gui_board,board,snake, food)
    end = board.check_end(snake)

    eat_self = snake.eat_self()
    update_snake(gui_board, board, direction, snake)
    draw_food(gui_board, food)
    board.update(snake, food)
    return end,eat_self

def draw_food(board,food:Food):
    py.draw.rect(board, cream, (food.x*swidth+1, food.y*sheight+1, swidth-1, sheight-1))
    py.display.update()


def check_eaten(gui_board,board, snake:Snake, food:Food):
    if snake.head[0] == food.x and snake.head[1] == food.y:
        #print(snake.points)
        #increase snakelength by 1
        grow_snake(gui_board,board,snake)
        #update food
        food.update()
        snake.ate_food = True

def end_message(board):
    font = py.font.Font('freesansbold.ttf', 32)
    # create a text surface object,
    # on which text is drawn on it.
    text = font.render('GAME OVER', True, red, grey)
    textRect = text.get_rect()
    # set the center of the rectangular object.
    textRect.center = (bwidth // 2, bheight // 2)
    board.blit(text, textRect)
    py.display.update()

def reset_to_start():
    pass


def get_output(snake:Snake, food:Food, dead:bool):
    #reinforcement function
    """
    define rewards for death, eating, surviving
    Return the reward to be input to the loss function for backprop
    """
    out = 0
    if dead:
        out = -1
        return out

    return out




def main():
    GUI()

if __name__ == '__main__':
    main()
            