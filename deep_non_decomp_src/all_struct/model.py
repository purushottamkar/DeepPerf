from blocks.bricks import MLP
from blocks.bricks import Tanh, Logistic, Rectifier, Linear
from blocks.initialization import IsotropicGaussian, Constant
from blocks.bricks.recurrent import LSTM
import numpy as np


class base_model(object):

    def __init__(self, x, input_dim):
        self.x = x
        self.input_dim = input_dim

    def create_model(self):
        x = self.x
        input_dim = self.input_dim
        mlp = MLP([Logistic(), Logistic(), Tanh()],
                  [input_dim, 100, 100, 1],
                  weights_init=IsotropicGaussian(0.001),
                  biases_init=Constant(0))
        mlp.initialize()
        self.mlp = mlp
        probs = mlp.apply(x)
        return probs


class recurrent_model(object):

    def __init__(self, x, input_dim, p):

        self.x = x
        self.input_dim = input_dim
        self.p = p

    def create_model(self):
        input_dim = self.input_dim
        x = self.x
        x_to_h = Linear(input_dim, input_dim * 4, name='x_to_h',
                        weights_init=IsotropicGaussian(),
                        biases_init=Constant(0.0))
        lstm = LSTM(input_dim, name='lstm',
                    weights_init=IsotropicGaussian(),
                    biases_init=Constant(0.0))
        h_to_o = Linear(input_dim, 1, name='h_to_o',
                        weights_init=IsotropicGaussian(),
                        biases_init=Constant(0.0))

        x_transform = x_to_h.apply(x)
        self.x_to_h = x_to_h
        self.lstm = lstm
        self.h_to_o = h_to_o

        h, c = lstm.apply(x_transform)

        # only values of hidden units of the last timeframe are used for
        # the classification
        probs = h_to_o.apply(h[-1])
        return probs
