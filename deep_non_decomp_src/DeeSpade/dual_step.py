import theano.tensor as T
import theano
import math


class MinTPRTNR(object):

    def __init__(self):
        pass

    def perf(self, TPR, TNR):
        return min(TPR, TNR)

    def dual_descent(self, alpha, beta):
        return alpha, beta

    def dual_projection(self, a, b):

        # Project onto straight line
        temp_a = (a - b + 1) / 2

        # Project onto first quadrant
        final_a = (temp_a <= 1.0) * (temp_a >= 0) * temp_a \
            + (temp_a > 1.0) * 1.0

        final_b = 1.0 - final_a

        return final_a, final_b

    def dual_step(self, y, alpha, beta, learning_rate, reward, flag, p):

        # The Dual descent  step
        a, b = self.dual_descent(alpha, beta)

        # Update the weight parameters
        temp_a = a - learning_rate * T.mean(y * reward.T)
        temp_b = b - learning_rate * T.mean((1 - y) * reward.T)

        # projection to sufficient dual region
        a, b = self.dual_projection(temp_a, temp_b)
        updates = [(alpha, a),
                   (beta, b)]
        return updates


class QMean(object):

    def __init__(self):
        pass

    def perf(self, TPR, TNR):
        return 1 - math.sqrt(((1 - TPR)**2 + (1 - TNR)**2) / 2.0)

    def dual_descent(self, alpha, beta, d_learning_rate):
        a = alpha - d_learning_rate
        b = beta - d_learning_rate
        return a, b

    def dual_projection(self, a, b):

        temp_a = (a > 0.0)
        temp_b = (b > 0.0)
        norm = a**2 + b**2
        in_side = (norm <= 0.5)

        final_a = (1 - in_side) * temp_a * a / (T.sqrt(norm)
                                                * 2.0) + in_side * temp_a * a
        final_b = (1 - in_side) * temp_b * b / (T.sqrt(norm)
                                                * 2.0) + in_side * temp_b * b

        return final_a, final_b

    def ftl_update(self, y, alpha, beta, reward):
        y = y.dimshuffle(0, 'x')
        cum_pos_rew = theano.shared(0.)
        cum_neg_rew = theano.shared(0.)
        cum_pos_rew_t = cum_pos_rew + T.sum(y * reward)
        cum_neg_rew_t = cum_neg_rew + T.sum((1 - y) * reward)
        alpha_t = cum_pos_rew_t**2 / (cum_pos_rew_t**2 + cum_neg_rew_t**2)
        beta_t = 1 - alpha_t
        updates = [(alpha, T.sqrt(alpha_t)),
                   (beta, T.sqrt(beta_t)),
                   (cum_pos_rew, cum_pos_rew_t),
                   (cum_neg_rew, cum_neg_rew_t)]
        return updates

    def ftl_update_TC(self, y, alpha, beta, reward, p=None):
        y = y.dimshuffle(0, 'x')
        cum_pos_rew = theano.shared(0.)
        cum_neg_rew = theano.shared(0.)
        cum_pos_rew_t = cum_pos_rew + \
            T.sum(y * (reward <= (0.5 / p)))
        cum_neg_rew_t = cum_neg_rew + \
            T.sum((1 - y) * (reward >= (0.5 / (1 - p))))
        alpha_t = cum_pos_rew_t / T.sqrt(cum_pos_rew_t**2 + cum_neg_rew_t**2)
        beta_t = T.sqrt(1 - alpha_t**2)
        updates = [(alpha, alpha_t),
                   (beta, beta_t),
                   (cum_pos_rew, cum_pos_rew_t),
                   (cum_neg_rew, cum_neg_rew_t)]
        return updates

    def dual_step(self, y, alpha, beta, learning_rate, reward,
                  ftl=False, p=None):

        if not ftl:
            # The Dual descent  step
            a, b = self.dual_descent(alpha, beta, learning_rate)

            # Update the weight parameters
            temp_a = a - learning_rate * T.mean(y * reward)
            temp_b = b - learning_rate * T.mean((1 - y) * reward)

            # projection to sufficient dual region
            a, b = self.dual_projection(temp_a, temp_b)
            updates = [(alpha, a),
                       (beta, b)]
        else:
            updates = self.ftl_update_TC(y, alpha, beta, reward, p)
        return updates
