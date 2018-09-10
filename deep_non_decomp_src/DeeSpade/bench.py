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
import math
theano.config.optimizer = 'None'
theano.config.exception_verbosity = 'high'


class BenchANN(object):

    def __init__(self, dual_class, num_epoch=10):
        self.num_epoch = num_epoch

    def create_model(self, x, y, input_dim, p):

        # Create the output of the MLP
        # mlp = MLP([Rectifier(), Rectifier(), Logistic()],
        #          [input_dim, 150, 100, 1],
        #          weights_init=IsotropicGaussian(0.01),
        #          biases_init=Constant(0))
        # mlp = MLP([Tanh(), Tanh(),  Tanh(), Logistic()],
        #          [input_dim, 60, 60, 80, 1],
        #          weights_init=IsotropicGaussian(0.001),
        #          biases_init=Constant(0))
        mlp = MLP([Rectifier(), Rectifier(), Logistic()],
                  [input_dim, 100, 80, 1],
                  weights_init=IsotropicGaussian(0.001),
                  biases_init=Constant(0))
        mlp.initialize()
        probs = mlp.apply(x)
        y = y.dimshuffle(0, 'x')

        cost = T.sum((y - probs)**2 * y * (1 - p) + (y - probs)**2 *
                     (1 - y) * p)
        # cost = T.sum((y - probs)**2)

        cost.name = "cost"

        return mlp, cost

    def primal_step(self, x, y, learning_rate, input_dim, p):

        mlp, cost = self.create_model(x, y, input_dim, p)
        cg = ComputationGraph([cost])
        weights = VariableFilter(roles=[WEIGHT])(cg.variables)
        updates = Adam(cost, weights)
        return mlp, updates,  cost

    def calc_cost(self, mlp, x, true_labels):
        pred_labels = mlp.apply(x)  # > 0.5
        return pred_labels

    def get_fns(self, input_dim=123,
                p_learning_rate=0.01,
                d_learning_rate=0.0001, p=0.23928176569346055):
        x = T.matrix('X')
        y = T.vector('y')

        mlp, primal_updates, loss = self.primal_step(x,
                                                     y,
                                                     p_learning_rate,
                                                     input_dim, p)
        updates = primal_updates

        train_fn = theano.function(
            [x, y], [loss], updates=updates)

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
            TPR = np.sum((pred_labels > 0.5) * 1.0 *
                         (Y_mat == 1)) / np.sum(Y_mat == 1)
            TNR = np.sum((pred_labels <= 0.5) * 1.0 *
                         (Y_mat == 0)) / np.sum(Y_mat == 0)
            print "TPR, TNR below"
            print TPR, TNR

            def Fmeasure(y, probs, thresh=0.9, beta=1):
                pred = probs > thresh
                print np.mean(pred == y)
                true_p = np.sum(pred * y) * 1.0
                false_p = np.sum(pred * (1 - y)) * 1.0
                false_n = np.sum((1 - pred) * y) * 1.0
                numerator = (1 + beta**2) * true_p
                denominator = (1 + beta**2) * true_p + \
                    beta**2 * false_n + false_p
                f_beta = numerator / denominator
                return f_beta

            def KLD(y_hat, y, tol=10e-8):
                P = (np.sum(y) + tol) / (2 * tol + y.shape[0])
                N = (np.sum(1 - y) + tol) / (2 * tol + y.shape[0])
                TPR = np.sum(y * (y_hat > 0.5)) / np.sum(y)
                TNR = np.sum((1 - y) * (y_hat <= 0.5)) / np.sum(1 - y)
                BA = (TPR + TNR) / 2
                PP = (np.sum(y_hat > 0.5) + tol) / (y.shape[0] + 2 * tol)
                NN = (np.sum(y_hat <= 0.5) + tol) / (y.shape[0] + 2 * tol)

                KLD = -P * np.log(P * 1.0 / PP) - \
                    N * np.log(N * 1.0 / NN)
                assert(abs(KLD) != np.inf), "KLD is zero" + str([P, N, PP, NN])

                return KLD, BA
            f_meas = Fmeasure(Y_mat, pred_labels)
            qmean = 1 - math.sqrt(((1 - TPR)**2 + (1 - TNR)**2) / 2.0)
            kld, ba = KLD(pred_labels, Y_mat)
            mintprtnr = min(TPR, TNR)
            return (mintprtnr, qmean, f_meas, kld, ba), pred_labels

        return train_fn, valid_fns


if __name__ == '__main__':
    X = np.random.randn(10, 5)
    y = np.asarray([0.,  0.,  1.,  0.,  1.,  1.,  0.,  1.,  0.,  1.])
    sm = BenchANN()
    train_fn, test_fn = sm.get_fns(input_dim=5)
