import theano
import theano.tensor as T
import numpy as np
import time
from model import base_model
from optimizer import Adam
from blocks.filter import VariableFilter
from blocks.graph import ComputationGraph
from blocks.roles import WEIGHT
import loss_functions
from c_code import mvc

theano.config.optimizer = 'None'
theano.config.exception_verbosity = 'high'


class struct_ann(object):

    def __init__(self):
        self.name = "base model"

    def get_scores(self, x, input_dim):
        self.model = base_model(x, input_dim)
        scores = self.model.create_model()
        return scores

    def calc_cost(self, mlp, x, true_labels):
        pred_labels = mlp.apply(x)  # > 0.5
        return pred_labels

    def get_fns(self, input_dim=123,
                p_learning_rate=0.01,
                loss='prec'):
        x = T.matrix('X')
        y = T.vector('y')
        loss_id = loss_functions.loss_dict[loss]
        scores = self.get_scores(x,
                                 input_dim)

        def get_score_fn():
            score_th_fn = theano.function(
                [x], [scores])
            return score_th_fn

        cost = T.sum((scores * y.dimshuffle(0, 'x')))
        cg = ComputationGraph([cost])
        weights = VariableFilter(roles=[WEIGHT])(cg.variables)
        updates = Adam(cost, weights)

        def get_update_fn():
            update_th_fn = theano.function([x, y], [cost], updates=updates)
            return update_th_fn

        score_th_fn = get_score_fn()

        def train_fn(x, y):
            y = 2 * y - 1
            update_th_fn = get_update_fn()
            scores = np.asarray(score_th_fn(x))
            beg = time.clock()
            Y_new = mvc.mvc(y.ravel(),
                            scores,
                            np.sum(y == 1), np.sum(y == -1),
                            loss_id)
            print "TIME TAKEN", str(time.clock() - beg)
            update_th_fn(x, Y_new)

        def valid_fns(x, y):
            y = 2 * y - 1
            scores = np.asarray(score_th_fn(x)).ravel()
            pred_labels = 2 * ((scores.ravel() > 0).astype(np.float32)) - 1
            loss_fn = loss_functions.get_loss_fn(loss_id)
            loss = loss_fn(y.ravel(), pred_labels)
            print np.sum(scores[y.ravel() == 1])
            print y.shape, np.sum(y), np.sum(pred_labels == 1), loss
            return loss
        return train_fn, valid_fns
