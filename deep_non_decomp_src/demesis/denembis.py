from blocks.bricks import MLP
from blocks.bricks import Tanh, Logistic, Rectifier, Linear
from blocks.filter import VariableFilter
from blocks.graph import ComputationGraph
from blocks.roles import WEIGHT, INPUT
from blocks.initialization import IsotropicGaussian, Constant
from optimizer import Adam
from blocks.bricks.recurrent import LSTM
import theano.tensor as T
import theano
import numpy as np
from models import base_model, recurrent_model

theano.config.optimizer = 'None'
theano.config.exception_verbosity = 'high'


class DeeNemBis(object):

    def __init__(self, dual_class, num_epoch=10, model='base', num_examples=60536):
        self.num_epoch = num_epoch
        self.dual_class = dual_class
        self.alpha = self.dual_class.alpha
        self.beta = self.dual_class.beta
        self.t = theano.shared(np.float32(0), name='time')
        self.model_type = model
        if model == 'base':
            self.model = base_model
        else:
            self.model = recurrent_model
        # np.float32(1.0/(num_examples * 10))
        dual_class.eps = np.float32(10e-8)

    def primal_step(self, x, y, learning_rate, input_dim, p):
        self.model = self.model(x, y, input_dim, p)
        score, probs = self.model.create_model()
        criterion = self.alpha * p - self.beta * np.float32(1 - p)

        r = theano.shared(np.float32(0.0), name='tp+fp')
        q = theano.shared(np.float32(0.0), name='tn+fn')

        pos_criterion = T.lt(probs, 0.5) * -criterion * score

        neg_criterion = T.gt(probs, 0.5) * criterion * score
        
        cost_weighed = T.mean(pos_criterion * T.gt(criterion, 0) +
                              neg_criterion * T.lt(criterion, 0))

        cg = ComputationGraph([cost_weighed])

        # Reward version
        r_temp = (self.t * r + T.mean(score * T.gt(probs, 0.5))) / (self.t + 1)
        q_temp = (self.t * q + T.mean(score * T.lt(probs, 0.5))) / (self.t + 1)

        
        # True Count version
        # r_temp = (self.t*r + T.mean(1.0 * T.gt(probs, 0.5)))/(self.t + 1)
        # q_temp = (self.t*q + T.mean(1.0 * T.lt(probs, 0.5)))/(self.t + 1)

        primal_updates = [(r, r_temp),
                          (q, q_temp),
                          (self.t, self.t + 1)]

        weights = VariableFilter(roles=[WEIGHT])(cg.variables)
        updates = Adam(cost_weighed, weights) + primal_updates

        # r = tp + fp
        # q = fp + fn

        primal_var = [r, q]
        return updates,  cost_weighed, score, primal_var

    def calc_cost(self, mlp, x, true_labels):
        pred_labels = mlp.apply(x)
        return pred_labels

    def get_fns(self, input_dim=123,
                p_learning_rate=0.01,
                d_learning_rate=0.0001, p=0.23928176569346055):
        x = T.matrix('X')
        y = T.vector('y')

        primal_updates, loss_weighed, \
            reward, primal_var = self.primal_step(x,
                                                  y,
                                                  p_learning_rate,
                                                  input_dim, p)
        [r, q] = primal_var
        dual_updates = self.dual_class.dual_updates(r=r, q=q)
        updates = primal_updates, dual_updates
        primal_train_fn = theano.function(
            [x, y], [r, self.alpha], updates=primal_updates,
            on_unused_input='ignore',
            name="Primal Train")
        dual_train_fn = theano.function(
            [], [self.alpha, self.beta], updates=dual_updates,
            name="Dual Train")

        def train_fn(x, y):
            r0_d, r1_d = primal_train_fn(x, y)
            alpha_d, beta_d = dual_train_fn()
            return r0_d, alpha_d

        # Calculate Validation in batch_mode for speedup
        x_mat = T.matrix('x_mat')
        y_mat = T.vector('y_mat')
        pred_labels = self.calc_cost(self.model, x_mat, y_mat)
        valid_th_fns = theano.function([x_mat], pred_labels)

        def valid_fns(X_mat, Y_mat):
            Y_mat = Y_mat.ravel()
            pred_labels = valid_th_fns(X_mat).ravel()
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
            return self.dual_class.perf(pred_labels, Y_mat), pred_labels
        return train_fn, valid_fns

if __name__ == '__main__':
    print "Sorry you need to run from the train file or \
take the pain of writing the main file."
