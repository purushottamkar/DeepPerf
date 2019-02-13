from blocks.bricks import MLP
from blocks.bricks import Tanh, Logistic, Rectifier
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


class Spade(object):

    def __init__(self, dual_class, num_epoch=10):
        self.num_epoch = num_epoch
        self.dual_class = dual_class

    def create_model(self, x, y, input_dim, p):

        # Create the output of the MLP
        mlp = MLP([Rectifier(), Rectifier(), Logistic()],
                  [input_dim, 150, 100, 1],
                  weights_init=IsotropicGaussian(0.01),
                  biases_init=Constant(0))
        mlp.initialize()
        probs = 1 - mlp.apply(x)
        y = y.dimshuffle(0, 'x')
        # Create the if-else cost function
        pos_ex = (y * probs) / p
        neg_ex = (1 - y) * (1 - probs) / np.float32(1 - p)
        reward = pos_ex + neg_ex
        cost = reward  # Negative of reward
        cost.name = "cost"

        return mlp, cost

    def primal_step(self, x, y, learning_rate, alpha, beta, input_dim, p):

        mlp, cost = self.create_model(x, y, input_dim, p)
        flag = T.eq(y, 1) * alpha + T.eq(y, 0) * beta
        cost_weighed = T.mean(cost * flag.dimshuffle(0, 'x'))
        cg = ComputationGraph([cost_weighed])
        weights = VariableFilter(roles=[WEIGHT])(cg.variables)
        updates = Adam(cost_weighed, weights)
        return mlp, updates,  cost_weighed, cost

    def calc_cost(self, mlp, x, true_labels):
        pred_labels = mlp.apply(x) > 0.5
        return pred_labels

    def get_fns(self, input_dim=123,
                p_learning_rate=0.01,
                d_learning_rate=0.00001, p=0.23928176569346055):
        alpha = theano.shared(0.5)
        beta = theano.shared(0.5)
        x = T.matrix('X')
        y = T.vector('y')

        mlp, primal_updates, loss_weighed, \
            reward = self.primal_step(x,
                                      y,
                                      p_learning_rate,
                                      alpha, beta,
                                      input_dim, p)
        dual_updates = self.dual_class.dual_step(y, alpha,
                                                 beta, d_learning_rate,
                                                 reward, True, p)
        updates = primal_updates + dual_updates

        train_fn = theano.function(
            [x, y], [reward, alpha, beta], updates=updates)

        # Calculate Validation in batch_mode for speedup
        x_mat = T.matrix('x_mat')
        y_mat = T.vector('y_mat')
        pred_labels = self.calc_cost(mlp, x_mat, y_mat)
        valid_th_fns = theano.function([x_mat], pred_labels)

        def valid_fns(X_mat, Y_mat):
            Y_mat = Y_mat.ravel()
            pred_labels = valid_th_fns(X_mat).ravel()
            # print np.sum(pred_labels == 0), np.sum(pred_labels == 1),
            # print np.sum(Y_mat == 1)
            TPR = np.sum((pred_labels == 1) * 1.0 *
                         (Y_mat == 1)) / np.sum(Y_mat == 1)
            TNR = np.sum((pred_labels == 0) * 1.0 *
                         (Y_mat == 0)) / np.sum(Y_mat == 0)
            # print "TPR, TNR below"
            # print TPR, TNR
            return self.dual_class.perf(TPR, TNR), pred_labels
        return train_fn, valid_fns

if __name__ == '__main__':
    X = np.random.randn(10, 5)
    y = np.asarray([0.,  0.,  1.,  0.,  1.,  1.,  0.,  1.,  0.,  1.])
    sm = Spade()
    train_fn, test_fn = sm.get_fns(input_dim=5)
