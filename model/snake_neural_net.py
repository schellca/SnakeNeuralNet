import torch
import torch.nn as nn
from torch import optim, Tensor
import numpy as np
import random
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')




class SnakeModel(nn.Module):
    """
    Convolutional model of the map of a game of snake
    Use convolutional layers to output a final classification decision of left, right, and straight
    """
    class ConvLayer(nn.Module):
        def __init__(self, input_size, output_size, kernel_size=3, stride=1, dropout=0.5):
            super().__init__()
            self.sequence = nn.Sequential(
                nn.Conv2d(input_size, output_size, kernel_size=kernel_size, padding=kernel_size // 2,stride=stride, bias=False),
                nn.LeakyReLU(),
                nn.MaxPool2d(kernel_size=2, stride=2),
                nn.Dropout(p=dropout)
            )

        def forward(self,z):
            return self.sequence(z)



    def __init__(self, layers=[32,64,96], input_size=32, n_input=3, n_output=4, kernel_size=3, stride=1, batch_size=128):
        super().__init__()
        self.batch_size = batch_size
        #initialize variables
        pooling_factor = 32 #2^number of layers
        pool_size = int(input_size/pooling_factor)
        self.n_output = n_output
        c = n_input
        self.n_conv = len(layers)
        
        for i, l in enumerate(layers):
            if i > 0:
                stride = 2
            self.add_module('conv%d' % i, self.ConvLayer(c, l, kernel_size, stride))
            c = l

        #Affine layers
        self.final = nn.Sequential(
            nn.Linear(in_features=l*pool_size*pool_size, out_features=128),
            nn.LeakyReLU(),
            nn.Linear(in_features=128, out_features=n_output),
            nn.LeakyReLU(),
            nn.Softmax()
        )
        
        


    def forward(self, z) -> nn.Module:
        for i in range(self.n_conv):
            z = self._modules['conv%d'%i](z)
            print(z.size())

        if z.dim() == 4:
            z = z.view(self.batch_size,-1)
        else:
            z = z.view(-1)
        return self.final(z)


    def predict(self, input) -> int:
        """
        Makes a direction prediction on the input
        :return: 0 thru 3 representing ahead, left, right
        """
        output = self.forward(input)
        prediction = torch.argmin(output)
        return prediction


def initialize_weights_kaiming(m):
    if isinstance(m, nn.Conv2d):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='leaky_relu')
        if m.bias is not None:
            nn.init.zeros_(m.bias)
    elif isinstance(m,nn.Linear):
        nn.init.kaiming_uniform_(m.weight, nonlinearity='leaky_relu')

def save_model(model):
    from torch import save
    from os import path
    return save(model.state_dict(), path.join(path.dirname(path.abspath(__file__)), 'snakenet.th'))

def load_model():
    from torch import load
    from os import path
    r = SnakeModel()
    r.load_state_dict(load(path.join(path.dirname(path.abspath(__file__)), 'snakenet.th'), map_location='cpu'))
    return r

#input sequence
"""
3 input channels
channel 1: empty squares
channel 2: snake squares
channel 3: food squares
"""

