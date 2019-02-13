from blocks.bricks import MLP
from blocks.bricks import Tanh, Logistic, Rectifier, Linear
from blocks.filter import VariableFilter
from blocks.graph import ComputationGraph, apply_dropout
from blocks.roles import WEIGHT, INPUT
from blocks.initialization import IsotropicGaussian, Constant, Orthogonal
from optimizer import Adam
from blocks.bricks.recurrent import LSTM
import theano.tensor as T
import theano
import numpy as np
from models import base_model, recurrent_model, recurrent_model_bench

theano.config.optimizer = 'fast_compile'
theano.config.exception_verbosity = 'high'


class DeeNemBis_bench(object):

    def __init__(self, dual_class, num_epoch=10, model='lstm', num_examples=60536):
        self.num_epoch = num_epoch
        self.dual_class = dual_class
        self.alpha = self.dual_class.alpha
        self.beta = self.dual_class.beta
        self.gamma = self.dual_class.gamma
        self.t = theano.shared(np.float32(0), name='time')
        self.model_type = model
        if model == 'base':
            self.model = base_model
        else:
            self.model = recurrent_model_bench
        # np.float32(1.0/(num_examples * 10))
        dual_class.eps = np.float32(10e-8)

    def primal_step(self, x, y, learning_rate, input_dim, p, mask=None):
        if mask is None:
            self.model = self.model(x, y, input_dim, p)
        else:
            self.model = self.model(x, y, input_dim, p, mask=mask)
        probs = self.model.create_model()
        cost = T.sum((probs - y.dimshuffle(0, 'x'))**2)
        cg = ComputationGraph([cost])

        weights = VariableFilter(roles=[WEIGHT])(cg.variables)

        updates = Adam(cost, weights)

        return updates, cost

    def calc_cost(self, mlp, x, true_labels, mask):
        pred_labels = mlp.apply(x, mask)
        return pred_labels

    def get_fns(self, input_dim=123,
                p_learning_rate=0.01,
                d_learning_rate=0.0001, p=0.23928176569346055):
        x = T.lmatrix('X')
        y = T.vector('y')
        m = T.lmatrix('mask_tr')
        updates, cost = self.primal_step(x,
                                         y,
                                         p_learning_rate,
                                         input_dim, p, mask=m)

        train_primal_fn = theano.function(
            [x, y, m], [], updates=updates,
            name="Primal Train")

        def train_fn(x, y, mask):
            train_primal_fn(x, y, mask.transpose())
        # Calculate Validation in batch_mode for speedup
        x_mat = T.lmatrix('x_mat')
        y_mat = T.vector('y_mat')
        mask_mat = T.lmatrix('mask_te')
        pred_labels = self.calc_cost(self.model, x_mat, y_mat, mask_mat)

        valid_th_fns = theano.function([x_mat, mask_mat], pred_labels)

        def valid_fns(X_mat, Y_mat, mask_mat, flag=0):
            Y_mat = Y_mat.ravel()
            pred_labels = valid_th_fns(X_mat, mask_mat).ravel()
            # print pred_labels, Y_mat
            # print np.sum(pred_labels == 0), np.sum(pred_labels == 1),
            # print np.sum(Y_mat == 1)
            # TPR = np.sum((pred_labels > 0.5) * 1.0 *
            #             (Y_mat == 1)) / np.sum(Y_mat == 1)
            # TNR = np.sum((pred_labels <= 0.5) * 1.0 *
            #             (Y_mat == 0)) / np.sum(Y_mat == 0)
            # print "TPR, TNR below"
            #P = np.mean(pred_labels)
            #N = np.mean(1 - pred_labels)
            # print TPR, TNR, np.sum(pred_labels), P, N
            return self.dual_class.perf(pred_labels, Y_mat, flag), pred_labels
        return train_fn, valid_fns

if __name__ == '__main__':
    print "Sorry you need to run from the train file or \
take the pain of writing the main file."
