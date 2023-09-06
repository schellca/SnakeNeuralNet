import torch
from utils.game_state import *

torch.set_printoptions(threshold=100000)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

def encode_board(snake:Snake, food:Food, board:Board):
    """
    takes board, snake and food as input
    returns 3 channel tensor for input to model
    """
    len_x = board.x
    len_y = board.y
    input_tensor = torch.zeros(3,len_x, len_y, dtype=torch.float32)
    input_tensor[0] = 1
    for i in range(len_y):
        for j in range(len_x):
            if [j,i] in snake.points:
                input_tensor[1,j,i] = 1
            elif [j,i] == [food.x,food.y]:
                input_tensor[2,j,i] = 1
            else:
                input_tensor[0,j,i] = 1

    return input_tensor.to(device)


def get_reward(snake:Snake, food:Food, new_snake:Snake):
    reward = 0
    old_dist = (snake.head[0]-food.x)**2 + (snake.head[1]-food.y)**2
    new_dist = (new_snake.head[0]-food.x)**2 + (new_snake.head[1]-food.y)**2
    if snake.dead == True:
        snake.dead = False
        reward = -10
    elif snake.ate_food:
        snake.ate_food = False
        reward =  10
    else:
        reward = 0.1
        """
    elif new_dist < old_dist:
        reward = 0.1
    else:
        reward = -0.1
    """
    return torch.tensor(reward)