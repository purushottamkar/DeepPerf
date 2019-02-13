from blocks.bricks import MLP
from blocks.bricks import Tanh, Logistic, Rectifier, Linear
from blocks.initialization import IsotropicGaussian, Constant, Orthogonal
from blocks.bricks.recurrent import LSTM
from blocks.bricks.lookup import LookupTable
import numpy as np
import theano.tensor as T

class base_model(object):

    def __init__(self, x, y, input_dim, p, mask=None):
        self.x = x
        self.y = y
        self.input_dim = input_dim
        self.p = p

    def create_model(self):
        x = self.x
        y = self.y
        input_dim = self.input_dim
        p = self.p
        mlp = MLP([Rectifier(), Rectifier(), Logistic()],
                  [input_dim, 100, 80, 1],
                  weights_init=IsotropicGaussian(0.001),
                  biases_init=Constant(0))
        mlp.initialize()
        self.mlp = mlp
        probs = mlp.apply(x)
        probs.name = "score"
        y = y.dimshuffle(0, 'x')
        # Create the if-else cost function
        pos_ex = (y * probs) / p
        neg_ex = (1 - y) * (1 - probs) / np.float32(1 - p)
        reward = pos_ex + neg_ex
        cost = reward  # Negative of reward
        cost.name = "cost"
        return cost, probs

    def apply(self, x):
        return self.mlp.apply(x)

class recurrent_model_bench(object):
    def __init__(self, x, y, input_dim, p, dict_size=9999, mask=None):

        self.x = x
        self.y = y
        self.input_dim = input_dim
        self.p = p
        self.mask = mask
        self.dict_size = dict_size
        self.embedding_dim = 512
        self.hidden_dim = 512
        
    def create_model(self):
        input_dim = self.input_dim
        x = self.x
        y = self.y
        p = self.p
        mask = self.mask
        hidden_dim = self.hidden_dim
        embedding_dim = self.embedding_dim
        lookup = LookupTable(self.dict_size, embedding_dim,
                             weights_init=IsotropicGaussian(0.001),
                             name='LookupTable')
        x_to_h = Linear(embedding_dim, hidden_dim * 4, name='x_to_h',
                        weights_init=IsotropicGaussian(0.001),
                        biases_init=Constant(0.0))
        lstm = LSTM(hidden_dim, name='lstm',
                    weights_init=IsotropicGaussian(0.001),
                    biases_init=Constant(0.0))
        h_to_o = MLP([Logistic()],
                     [hidden_dim, 1],
                     weights_init=IsotropicGaussian(0.001),
                     biases_init=Constant(0),
                     name='h_to_o')
        
        lookup.initialize()
        x_to_h.initialize()
        lstm.initialize()
        h_to_o.initialize()
        
        embed = lookup.apply(x).reshape((x.shape[0],
                                        x.shape[1],
                                        self.embedding_dim))
        embed.name = "embed_vec"
        x_transform = x_to_h.apply(embed.transpose(1, 0, 2))
        x_transform.name = "Transformed X"
        self.lookup = lookup
        self.x_to_h = x_to_h
        self.lstm = lstm
        self.h_to_o = h_to_o

        #if mask is None:
        h, c = lstm.apply(x_transform)
        #else:
        #h, c = lstm.apply(x_transform, mask=mask)
        h.name = "hidden_state"
        c.name = "cell state"
        # only values of hidden units of the last timeframe are used for
        # the classification
        indices = T.sum(mask, axis=0) - 1
        rel_hid = h[indices, T.arange(h.shape[1])]
        out = self.h_to_o.apply(rel_hid)
        
        probs = out
        return probs

    def apply(self, x, mask=None):
        embed = self.lookup.apply(x)
        x_transform = self.x_to_h.apply(embed.transpose(1, 0, 2))
        h, c = self.lstm.apply(x_transform)
        indices = T.sum(mask, axis=1) - 1
        rel_hid = h[ indices, T.arange(h.shape[1])]
        out = self.h_to_o.apply(rel_hid)
        return out

class recurrent_model(object):

    def __init__(self, x, y, input_dim, p, dict_size=9999, mask=None):

        self.x = x
        self.y = y
        self.input_dim = input_dim
        self.p = p
        self.mask = mask
        self.dict_size = dict_size
        self.embedding_dim = 64
        self.hidden_dim = 16
        
    def create_model(self):
        input_dim = self.input_dim
        x = self.x
        y = self.y
        p = self.p
        mask = self.mask
        hidden_dim = self.hidden_dim
        embedding_dim = self.embedding_dim
        lookup = LookupTable(self.dict_size, embedding_dim,
                             weights_init=IsotropicGaussian(0.001),
                             name='LookupTable')
        x_to_h = Linear(embedding_dim, hidden_dim * 4, name='x_to_h',
                        weights_init=IsotropicGaussian(0.001),
                        biases_init=Constant(0.0))
        lstm = LSTM(hidden_dim, name='lstm',
                    weights_init=IsotropicGaussian(0.001),
                    biases_init=Constant(0.0))
        h_to_o = MLP([Logistic()],
                     [hidden_dim, 1],
                     weights_init=IsotropicGaussian(0.001),
                     biases_init=Constant(0),
                     name='h_to_o')
        
        lookup.initialize()
        x_to_h.initialize()
        lstm.initialize()
        h_to_o.initialize()
        
        embed = lookup.apply(x).reshape((x.shape[0],
                                        x.shape[1],
                                        self.embedding_dim))
        embed.name = "embed_vec"
        x_transform = x_to_h.apply(embed.transpose(1, 0, 2))
        x_transform.name = "Transformed X"
        self.lookup = lookup
        self.x_to_h = x_to_h
        self.lstm = lstm
        self.h_to_o = h_to_o

        #if mask is None:
        h, c = lstm.apply(x_transform)
        #else:
        #h, c = lstm.apply(x_transform, mask=mask)
        h.name = "hidden_state"
        c.name = "cell state"
        # only values of hidden units of the last timeframe are used for
        # the classification
        indices = T.sum(mask, axis=0) - 1
        rel_hid = h[indices, T.arange(h.shape[1])]
        out = self.h_to_o.apply(rel_hid)
        
        probs = 1 - out
        probs.name = "probability"
        y = y.dimshuffle(0, 'x')
        # Create the if-else cost function
        pos_ex = (y * probs) / p
        neg_ex = (1 - y) * (1 - probs) / np.float32(1 - p)
        reward = pos_ex + neg_ex
        cost = reward  # Negative of reward
        cost.name = "cost"
        return cost

    def apply(self, x, mask=None):
        embed = self.lookup.apply(x)
        x_transform = self.x_to_h.apply(embed.transpose(1, 0, 2))
        h, c = self.lstm.apply(x_transform)
        indices = T.sum(mask, axis=1) - 1
        rel_hid = h[ indices, T.arange(h.shape[1])]
        out = self.h_to_o.apply(rel_hid)
        return 1.0 - out
