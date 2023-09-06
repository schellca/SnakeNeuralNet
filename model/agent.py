import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from utils.game_state import *
from utils.encoder_decoder import *
from model.snake_neural_net import *
from collections import deque
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
print(device)





class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return state, action, reward, next_state, done

    def __len__(self):
        return len(self.buffer)


class Agent:
    def __init__(self, lr = 1e-4, gamma = 0.99, load=False):
        self.load = load
        if load:
            self.model = load_model()
            self.model = self.model.to(device)
        else:
            self.model = SnakeModel().to(device)
        self.model.apply(initialize_weights_kaiming)
        self.optimizer = torch.optim.Adam(self.model.parameters(),lr=lr)
        self.gamma=gamma

    def get_action(self, state, epsilon):
        if self.load:
            q_values = self.model(state)
            print(q_values)
            return torch.argmax(q_values).item()
        else:
            if random.random() < epsilon:
                return random.choice([i for i in range(4)])
            else:
                q_values = self.model(state)
                return torch.argmax(q_values).item()
        
    def train(self, buffer:ReplayBuffer, batch_size):
        state, action, reward, next_state, done = buffer.sample(batch_size)
        action = torch.LongTensor(action).to(device)
        reward = torch.FloatTensor(reward).to(device)
        next_state = torch.stack(next_state).to(device)
        done = torch.FloatTensor(done).to(device)
        state = torch.stack(state).to(device)
        

        
        

        q_values = self.model.forward(state)
        next_q_values = self.model.forward(next_state)
        print(q_values)
        print('\n')
        print(next_q_values)

        q_value = q_values.gather(1, action.unsqueeze(1)).squeeze(1)
        next_q_value = next_q_values.max(1)[0]
        
        target_q_value = reward + self.gamma*next_q_value*(1-done)
        
        loss = F.mse_loss(q_value, target_q_value.detach())
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        print(loss.item(), 'loss')


    def return_model(self):
        return self.model
