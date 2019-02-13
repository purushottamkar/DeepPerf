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
from models import base_model, recurrent_model

theano.config.optimizer = 'fast_compile'
theano.config.exception_verbosity = 'high'


class DeeNemBis(object):

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
            self.model = recurrent_model
        # np.float32(1.0/(num_examples * 10))
        dual_class.eps = np.float32(10e-8)

    def primal_step(self, x, y, learning_rate, input_dim, p, mask=None):
        if mask is None:
            self.model = self.model(x, y, input_dim, p)
        else:
            self.model = self.model(x, y, input_dim, p, mask=mask)
        cost = self.model.create_model()

        flag = T.eq(y, 1) * (self.gamma[0] * self.alpha[0] +
                             self.gamma[1] * self.beta[0]) +\
            T.eq(y, 0) * (self.gamma[0] * self.alpha[1] +
                          self.gamma[1] * self.beta[0])

        q0 = theano.shared(np.float32(0), name='q0')
        q1 = theano.shared(np.float32(0), name='q1')
        r0 = theano.shared(np.float32(0), name='r0')
        r1 = theano.shared(np.float32(0), name='r1')

        q0_temp = q0 * self.t + T.mean((T.eq(y, 1) * self.alpha[0] +
                                        T.eq(y, 0) *
                                        self.alpha[1]).dimshuffle(0, 'x') *
                                       cost)
        q1_temp = q1 * self.t + T.mean((T.eq(y, 1) * self.beta[0] +
                                        T.eq(y, 0) *
                                        self.beta[1]).dimshuffle(0, 'x') *
                                       cost)
        # Update r
        r0_next = (r0 * self.t + T.mean(T.eq(y, 1).dimshuffle(0, 'x') *
                                        cost)) * 1.0 / (self.t + 1)
        r1_next = (r1 * self.t + T.mean(T.eq(y, 0).dimshuffle(0, 'x') *
                                        cost)) * 1.0 / (self.t + 1)

        # Update q
        q0_next = (q0_temp -
                   self.dual_class.dual1_fn(self.alpha)) / (self.t + 1)
        q1_next = (q1_temp -
                   self.dual_class.dual2_fn(self.beta)) / (self.t + 1)

        primal_updates = [(q0, q0_next),
                          (q1, q1_next),
                          (r0, r0_next),
                          (r1, r1_next),
                          (self.t, self.t + 1)]

        cost_weighed = T.mean(cost * flag.dimshuffle(0, 'x'))
        cg = ComputationGraph([cost_weighed])

        weights = VariableFilter(roles=[WEIGHT])(cg.variables)
        
        updates = Adam(cost_weighed, weights) + primal_updates

        primal_var = [[r0, r1], [q0, q1]]
        return updates,  cost_weighed, cost, primal_var

    def calc_cost(self, mlp, x, true_labels, mask):
        pred_labels = mlp.apply(x, mask)
        return pred_labels

    def save_emb(self):
        return self.model.lookup.W.get_value()
    
    def get_fns(self, input_dim=123,
                p_learning_rate=0.01,
                d_learning_rate=0.0001, p=0.23928176569346055):
        x = T.lmatrix('X')
        y = T.vector('y')
        m = T.lmatrix('mask_tr')
        primal_updates, loss_weighed, \
            reward, primal_var = self.primal_step(x,
                                                  y,
                                                  p_learning_rate,
                                                  input_dim, p, mask=m)
        [r, q] = primal_var
        dual_updates = self.dual_class.dual_updates(r=r, q=q)
        updates = primal_updates, dual_updates
        pu, du = updates

        primal_train_fn = theano.function(
            [x, y, m], [r[0], self.alpha[0]], updates=primal_updates,
            name="Primal Train")
        dual_train_fn = theano.function(
            [], [self.alpha[0], self.beta[0]], updates=dual_updates,
            name="Dual Train")

        def train_fn(x, y, mask):
            r0_d, r1_d = primal_train_fn(x, y, mask.transpose())
            alpha_d, beta_d = dual_train_fn()
            return alpha_d, beta_d

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
