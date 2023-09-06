import numpy as np
import copy
import torch
torch.set_printoptions(threshold=100000)

class Food:
    def __init__(self,x_sq,y_sq):
        self.xsq = x_sq
        self.ysq = y_sq
        self.x = np.random.randint(0,self.xsq-1)
        self.y = np.random.randint(0,self.ysq-1)
        self.consumed = False

    def update(self):
        #randomly update food position on board
        self.x = np.random.randint(0,self.xsq-1)
        self.y = np.random.randint(0,self.ysq-1)

    def reset(self):
        self.x = np.random.randint(0,self.xsq-1)
        self.y = np.random.randint(0,self.ysq-1)
        self.consumed = False

class Snake:
    def __init__(self, length:int, board_size:tuple):
        #randomly initialize direction
        self.move_dir = np.random.randint(0,3)
        self.tail_dir = self.move_dir
        self.points = [[0,0] for _ in range(length)]
        
        
        self.xbound = board_size[0]
        self.ybound = board_size[1]
        self.set()
        self.dead = False
        self.ate_food = False
        print(self.points)

    def reset(self):
        move_dir = np.random.randint(0,3)
        self.set(move_dir = move_dir)

    def set(self, move_dir=None):
        if move_dir != None:
            self.move_dir = move_dir
        if self.move_dir == 0:
            #left
            for i,x in enumerate(self.points):
                self.points[i] = [int(self.xbound/2)+i,int(self.ybound/2)]
                
        elif self.move_dir == 2:
            #right
            for i,x in enumerate(self.points):
                self.points[i] = [int(self.xbound/2)-i,int(self.ybound/2)]
                
        elif self.move_dir == 1:
            #up
            for i,x in enumerate(self.points):
                self.points[i] = [int(self.xbound/2),int(self.ybound/2)+i]
                
        elif self.move_dir == 3:
            #down
            for i,x in enumerate(self.points):
                self.points[i] = [int(self.xbound/2),int(self.ybound/2)-i]
        self.head = self.points[0]
        self.tail = self.points[1:]
                
    
    def update(self, move_dir=None):
        if move_dir != None:
            self.move_dir = move_dir
        temp1 = copy.deepcopy(self.points[0])
        for i in range(len(self.points)):
            if i != 0:
                temp2 = copy.deepcopy(self.points[i])
                self.points[i] = temp1
                temp1 = temp2

        if move_dir == 0:
            #left
            self.points[0][0]-=1
        elif move_dir == 2:
            #right
            self.points[0][0]+=1
        elif move_dir == 1:
            #up
            self.points[0][1]-=1
        elif move_dir == 3:
            #down
            self.points[0][1]+=1
        
        self.head = self.points[0]
        self.tail = self.points[1:]
    
    def grow(self):
        point = self.get_tail_dir()
        if self.tail_dir == 0:
            #moving left, spawn right
            self.points.append([point[0]+1, point[1]])
        elif self.tail_dir == 2:
            #moving right, spawnleft
            self.points.append([point[0]-1, point[1]])
        elif self.tail_dir == 1:
            #moving up, spawn down
            self.points.append([point[0], point[1]+1])
        elif self.tail_dir == 3:
            #moving down, spawn up
            self.points.append([point[0], point[1]-1])

    def get_tail_dir(self):
        point1 = self.points[-2]
        point2 = self.points[-1]

        if point1[0] > point2[0]:
            #moving right
            self.tail_dir = 2
        elif point1[0]<point2[0]:
            #moving left
            self.tail_dir = 0
        elif point1[1]>point2[1]:
            #moving down
            self.tail_dir = 3
        elif point1[1]<point2[1]:
            #moving up
            self.tail_dir = 1

        return point2
    
    def eat_self(self):
        for s in self.tail:
            if self.head[0] == s[0] and self.head[1] == s[1]:
                #print(self.head, self.tail)
                return True
        return False
    
    def new_dir(self, direction):
        if self.move_dir != direction:
            #moving left
            if direction == self.move_dir+1 or direction == self.move_dir-3:
                #right turn
                self.move_dir = direction
            elif direction == self.move_dir-1 or direction == self.move_dir+3:
                #left turn
                self.move_dir = direction





class Board:
    def __init__(self, x_sq, y_sq):
        self.x = x_sq
        self.y = y_sq
        self.matrix = np.zeros((y_sq,x_sq))

    def update(self, snake: Snake, food: Food):
        try:
            self.matrix = np.zeros((self.y,self.x))
            self.matrix[food.y,food.x] = 2
            for x in snake.points:
                self.matrix[x[1],x[0]] = 1
        except:
            pass

    def check_end(self, snake:Snake):
        if snake.head[0] <= -1 and snake.move_dir == 0:
            snake.dead = True
            return True
        elif snake.head[0] >= self.x and snake.move_dir == 2:
            snake.dead = True
            return True
        elif snake.head[1] <= -1 and snake.move_dir == 1:
            snake.dead = True
            return True
        elif snake.head[1] >= self.y and snake.move_dir == 3:
            snake.dead = True
            return True
        else:
            return False
    
    def reset(self):
        self.matrix = np.zeros((self.y,self.x))


"""
def update_metrics(x,y,x_food,y_food, snake, direction):

    #get wall distances

    #check if snake is in path
    t_above, t_below, t_right, t_left = check_in_path(snake, direction)
"""
    



