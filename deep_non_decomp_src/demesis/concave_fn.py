import theano.tensor as T
import numpy as np
import theano
theano.config.traceback.limit = 10
floatX = theano.config.floatX

'''__author__: Amartya Sanyal <amartya18x@gmail.com>'''


class BaKLD(object):

    def __init__(self, C):
        self.C = np.float32(C)
        self.alpha = (theano.shared(np.float32(0.5), name='alpha0'),
                      theano.shared(np.float32(0.5), name='alpha1'))
        self.beta = (theano.shared(np.float32(0.0), name='beta0'),
                     theano.shared(np.float32(0.0), name='beta1'))
        self.gamma = (theano.shared(np.float32(C), name='gamma0'),
                      theano.shared(np.float32(1 - C), name='gamma1'))
        self.eps = 10e-8

    def dual_updates(self, r=None, q=None):
        z = r[1] - r[0]
        alpha_upd = [(self.alpha[0], self.alpha[0]),
                     (self.alpha[1], self.alpha[1])]
        beta_upd = [(self.beta[0], np.float32(2) * z),
                    (self.beta[1],  -np.float32(2) * z)]
        gamma_upd = [(self.gamma[0], self.gamma[0]),
                     (self.gamma[1], self.gamma[1])]
        dual_updates = alpha_upd + beta_upd + gamma_upd
        return dual_updates

    def smooth(self, p, tol=2):
        tol = self.eps

        return (p + tol) / (2.0 * tol + 1.0)

    def perf(self, y_hat, y):
        tol = self.eps
        P = (np.sum(y)) / (y.shape[0] * 1.0)
        N = (np.sum(1 - y)) / (y.shape[0] * 1.0)
        PP = (np.sum(y_hat > 0.5)) / (y.shape[0] * 1.0)
        NN = (np.sum(y_hat <= 0.5)) / (y.shape[0] * 1.0)

        TPR = np.sum(y * (y_hat > 0.5)) / np.sum(y)
        TNR = np.sum((1 - y) * (y_hat <= 0.5)) / np.sum(1 - y)
        BA = (TPR + TNR) / 2

        PS = self.smooth(P)
        NS = self.smooth(N)
        PPS = self.smooth(PP)
        NNS = self.smooth(NN)
        KLD = -PS * np.log(PS * 1.0 / PPS) - \
            NS * np.log(NS * 1.0 / NNS)
        assert(abs(KLD) != np.inf), "KLD is zero" + str([P, N, PP, NN])
        # print P, PS, N, NS, PP, PPS, NN, NNS, KLD, BA
        return (BA, KLD, self.eval_cover(BA, KLD))

    def eval_perf(self, y_hat, y):
        perf1 = self.eval_meas_1(y_hat, y)
        perf2 = self.eval_meas_2(y_hat, y)
        perf = self.eval_cover(perf1, perf2)
        return perf

    def eval_meas_1(self, y_hat, y):
        TPR = T.sum(T.eq(y, 1) * y_hat) / T.sum(T.eq(y, 1))
        TNR = T.sum(T.eq(y, 0) * (1 - y_hat)) / T.sum(T.eq(y, 0))
        return (TPR + TNR) / 2

    def eval_meas_2(self, y, y_hat):
        batch_size = y.shape[0]
        P = T.sum(y, 1) / batch_size
        N = T.sum(y, 0) / batch_size
        TPR = T.sum(T.eq(y, 1) * y_hat) / batch_size
        TNR = T.sum(T.eq(y, 0) * (1 - y_hat)) / batch_size
        return P * T.log(P * 1.0 / TPR) + N * T.log(N * 1.0 / TNR)

    def eval_cover(self, eval1, eval2):
        return self.C * eval1 + (1 - self.C) * eval2

    def dual1_fn(self, alpha):
        (r0, r1) = alpha
        min_val = T.le(r0, r1) * r0 + T.lt(r1, r0) * r1
        return min_val

    def dual2_fn(self, beta):
        (r0, r1) = beta
        min_val = T.le(r0, r1) * r0 + T.lt(r1, r0) * r1
        return min_val

    def dual_cov(self, q0, q1):
        return q0, q1


class KLD(object):

    def __init__(self, C):
        # C is the proportion of positives
        self.C = np.float32(C)
        self.alpha = theano.shared(np.float32(0.0), name='alpha0')
        self.beta = theano.shared(np.float32(1.0), name='beta0')
        self.eps = 10e-8

    def smooth(self, p, tol=2):
        tol = self.eps

        return (p + tol) / (2.0 * tol + 1.0)

    def dual_updates(self, r=None, q=None):

        alpha_upd = [(self.alpha, 1.0 / (r + self.eps))]
        beta_upd = [(self.beta, 1.0 / (q + self.eps))]
        dual_updates = alpha_upd + beta_upd
        return dual_updates

    def perf(self, y_hat, y, flag=1):
        P = (np.sum(y)) / (y.shape[0] * 1.0)
        N = (np.sum(1 - y)) / (y.shape[0] * 1.0)
        PP = (np.sum(y_hat > 0.5)) / (y.shape[0] * 1.0)
        NN = (np.sum(y_hat <= 0.5)) / (y.shape[0] * 1.0)

        PS = self.smooth(P)
        NS = self.smooth(N)
        PPS = self.smooth(PP)
        NNS = self.smooth(NN)
        KLD = -PS * np.log(PS * 1.0 / PPS) - \
            NS * np.log(NS * 1.0 / NNS)
        assert(abs(KLD) != np.inf), "KLD is zero" + str([P, N, PP, NN])
        if flag:
            print P, PS, N, NS, PP,  NN,
            y.shape[0], np.sum(y_hat > 0.5), np.sum(y_hat <= 0.5), KLD
            print y_hat.shape, y.shape
        return KLD

    def eval_perf(self, y_hat, y):
        perf1 = self.eval_meas_1(y_hat, y)
        perf2 = self.eval_meas_2(y_hat, y)
        perf = self.eval_cover(perf1, perf2)
        return perf

    def eval_meas_1(self, y_hat, y):
        TPR = T.sum(T.eq(y, 1) * y_hat) / T.sum(T.eq(y, 1))
        TNR = T.sum(T.eq(y, 0) * (1 - y_hat)) / T.sum(T.eq(y, 0))
        return (TPR + TNR) / 2

    def eval_meas_2(self, y, y_hat):
        batch_size = y.shape[0]
        P = T.sum(y, 1) / batch_size
        N = T.sum(y, 0) / batch_size
        TPR = T.sum(T.eq(y, 1) * y_hat) / batch_size
        TNR = T.sum(T.eq(y, 0) * (1 - y_hat)) / batch_size
        return P * T.log(P * 1.0 / TPR) + N * T.log(N * 1.0 / TNR)

    def eval_cover(self, eval1, eval2):
        return self.C * eval1 + (1 - self.C) * eval2

    def dual1_fn(self, alpha):
        (r0, r1) = alpha
        min_val = T.le(r0, r1) * r0 + T.lt(r1, r0) * r1
        return min_val

    def dual2_fn(self, beta):
        (r0, r1) = beta
        min_val = T.le(r0, r1) * r0 + T.lt(r1, r0) * r1
        return min_val

    def dual_cov(self, q0, q1):
        return q0, q1
