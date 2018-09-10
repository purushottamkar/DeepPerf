from blocks.bricks import MLP
from blocks.bricks import Tanh, Logistic, Rectifier
from theano.compile.nanguardmode import NanGuardMode
from blocks.filter import VariableFilter
from blocks.graph import ComputationGraph
from blocks.roles import WEIGHT
from blocks.initialization import IsotropicGaussian, Constant
from optimizer import Adam
import theano.tensor as T
import theano
import numpy as np

theano.config.optimizer = 'None'
theano.config.exception_verbosity = 'high'


class FbetaOpt(object):

    def __init__(self, p, beta=1, num_epoch=10):
        self.num_epoch = num_epoch
        self.beta = beta
        self.p = p

    def create_model(self, x, y, input_dim, tol=10e-5):

        # Create the output of the MLP
        mlp = MLP([Rectifier(), Rectifier(), Logistic()],
                  [input_dim, 100, 100, 1],
                  weights_init=IsotropicGaussian(0.01),
                  biases_init=Constant(0))
        mlp.initialize()
        probs = mlp.apply(x)
        y = y.dimshuffle(0, 'x')
        # Create the if-else cost function
        true_p = (T.sum(y * probs) + tol) * 1.0 / (T.sum(y) + tol)
        true_n = (T.sum((1 - y) * (1 - probs)) + tol) * \
            1.0 / (T.sum(1 - y) + tol)
        #p = (T.sum(y) + tol) / (y.shape[0] + tol)
        theta = (1 - self.p) / self.p
        numerator = (1 + self.beta**2) * true_p
        denominator = self.beta**2 + theta + true_p - theta * true_n

        Fscore = numerator / denominator

        cost = -1 * Fscore
        cost.name = "cost"

        return mlp, cost, probs

    def primal_step(self, x, y, learning_rate, input_dim):

        mlp, cost, probs = self.create_model(x, y, input_dim)
        cg = ComputationGraph([cost])
        weights = VariableFilter(roles=[WEIGHT])(cg.variables)
        updates = Adam(cost, weights)
        return mlp, updates,  cost, probs

    def get_cost(self, y, probs, thresh=0.5):
        pred = probs > thresh
        true_p = np.sum(pred * y) * 1.0
        false_p = np.sum(pred * (1 - y)) * 1.0
        false_n = np.sum((1 - pred) * y) * 1.0
        numerator = (1 + self.beta**2) * true_p
        denominator = (1 + self.beta**2) * true_p + \
            self.beta**2 * false_n + false_p

        f_beta = numerator / denominator
        return f_beta

    def get_fns(self, input_dim=123,
                p_learning_rate=0.01,
                d_learning_rate=0.0001, p=0.23928176569346055):
        x = T.matrix('X')
        y = T.vector('y')

        mlp, updates, cost, probs = self.primal_step(x, y,
                                                     p_learning_rate,
                                                     input_dim)
        train_fn = theano.function(
            [x, y], [cost], updates=updates,
            mode=NanGuardMode(nan_is_error=True,
                              inf_is_error=True,
                              big_is_error=True)
        )

        # Calculate Validation in batch_mode for speedup
        valid_th_fns = theano.function([x], probs)

        def valid_fn(x, y):
            probs = valid_th_fns(x)
            f_beta = self.get_cost(y, probs)
            return f_beta
        return train_fn, valid_fn

if __name__ == '__main__':
    X = np.random.randn(10, 5)
    y = np.asarray([0.,  0.,  1.,  0.,  1.,  1.,  0.,  1.,  0.,  1.])
    sm = FbetaOpt()
    train_fn, test_fn = sm.get_fns(input_dim=5)
