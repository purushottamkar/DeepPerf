from blocks.bricks import MLP
from blocks.bricks import Tanh, Logistic
from blocks.filter import VariableFilter
from blocks.graph import ComputationGraph
from blocks.roles import WEIGHT
from blocks.initialization import IsotropicGaussian, Constant
from optimizer import Adam
import theano.tensor as T
import theano
import numpy as np

theano.config.optimizer = 'None'


class Spade(object):

    def __init__(self, num_epoch=10):
        self.num_epoch = num_epoch

    def create_model(self, x, y, input_dim, p):

        # Create the output of the MLP
        mlp = MLP([Tanh(), Tanh(), Logistic()], [input_dim, 200, 100, 1],
                  weights_init=IsotropicGaussian(0.01),
                  biases_init=Constant(0))
        mlp.initialize()
        probs = mlp.apply(x).sum()

        # Create the if-else cost function
        reward = (probs * y * 1.0) / p + (1 - probs) * (1 - y) * 1.0 / (1 - p)
        cost = -reward  # Negative of reward
        cost.name = "cost"

        return mlp, cost

    def dual_descent(self, alpha, beta):
        return alpha, beta

    def dual_projection(self, a, b):

        # Project onto straight line
        temp_a = (b - a + 1) / 2
        temp_b = (a - b + 1) / 2

        # Project onto first quadrant
        final_a = (temp_a <= 1.0) * (temp_a >= 0) * temp_a \
            + (temp_a > 1.0) * 1.0

        final_b = (temp_b <= 1.0) * (temp_b >= 0) * temp_b \
            + (temp_b > 1.0) * 1.0

        return final_a, final_b

    def dual_step(self, y, alpha, beta, learning_rate, reward):

        # The Dual descent  step
        a, b = self.dual_descent(alpha, beta)

        # Update the weight parameters
        temp_a = a - y * learning_rate * reward
        temp_b = b - (1 - y) * learning_rate * reward

        # projection to sufficient dual region
        a, b = self.dual_projection(temp_a, temp_b)
        updates = [(alpha, a),
                   (beta, b)]
        return updates

    def primal_step(self, x, y, learning_rate, alpha, beta, input_dim, p):

        mlp, cost = self.create_model(x, y, input_dim, p)
        cg = ComputationGraph([cost])
        weights = VariableFilter(roles=[WEIGHT])(cg.variables)
        updates = Adam(cost, weights, y, alpha, beta)
        return mlp, updates, -1 * cost

    def calc_cost(self, mlp, x, true_labels):
        pred_labels = mlp.apply(x) > 0.5
        return pred_labels

    def get_fns(self, input_dim=123,
                p_learning_rate=0.001,
                d_learning_rate=0.001, p=0.23928176569346055):
        alpha = theano.shared(0.)
        beta = theano.shared(0.)
        x = T.vector('X')
        y = T.scalar('y')

        mlp, primal_updates, reward = self.primal_step(x, y,
                                                       p_learning_rate,
                                                       alpha, beta,
                                                       input_dim, p)
        dual_updates = self.dual_step(y, alpha, beta, d_learning_rate, reward)
        updates = primal_updates + dual_updates

        train_fn = theano.function([x, y], reward, updates=updates)

        # Calculate Validation in batch_mode for speedup
        x_mat = T.matrix('x_mat')
        y_mat = T.vector('y_mat')
        pred_labels = self.calc_cost(mlp, x_mat, y_mat)
        valid_th_fns = theano.function([x_mat], pred_labels)

        def valid_fns(X_mat, Y_mat):
            Y_mat = Y_mat.ravel()
            pred_labels = valid_th_fns(X_mat).ravel()
            #print np.sum(pred_labels)
            #print np.sum(pred_labels == 0), np.sum(pred_labels == 1),
            #print np.sum(Y_mat == 1)
            TPR = np.sum((pred_labels == 1) * 1.0 *
                         (Y_mat == 1)) / np.sum(Y_mat == 1)
            TNR = np.sum((pred_labels == 0) * 1.0 *
                         (Y_mat == 0)) / np.sum(Y_mat == 0)
            #print "TPR, TNR below"
            #print TPR, TNR
            return min(TPR, TNR), pred_labels
        return train_fn, valid_fns

if __name__ == '__main__':
    X = np.random.randn(10, 5)
    y = np.asarray([0.,  0.,  1.,  0.,  1.,  1.,  0.,  1.,  0.,  1.])
    sm = Spade()
    train_fn, test_fn = sm.get_fns(input_dim=5)
