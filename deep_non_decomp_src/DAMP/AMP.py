from blocks.bricks import MLP
from blocks.bricks import Tanh, Logistic, Rectifier
from blocks.bricks.cost import SquaredError, MisclassificationRate, BinaryCrossEntropy
from theano.compile.nanguardmode import NanGuardMode
from blocks.filter import VariableFilter
from blocks.graph import ComputationGraph, apply_dropout
from blocks.roles import WEIGHT, INPUT
from blocks.initialization import IsotropicGaussian, Constant
from blocks.extensions.monitoring import DataStreamMonitoring
from blocks.main_loop import MainLoop
from blocks.extensions import FinishAfter, Printing
from blocks_extras.extensions.plot import Plot
from blocks.algorithms import GradientDescent, Adam
import theano.tensor as T
import theano
import numpy as np

theano.config.optimizer = 'None'
theano.config.exception_verbosity = 'high'


class FbetaThresh(object):

    def __init__(self, p, beta=1, num_epoch=10):
        self.num_epoch = num_epoch
        self.beta = beta
        self.p = p
        self.theta = (1 - p) * 1.0 / p

    def create_base_model(self, x, y, input_dim):

        # Create the output of the MLP
        mlp = MLP([Tanh(), Tanh(), Logistic()],
                  [input_dim, 100, 100, 1],
                  weights_init=IsotropicGaussian(0.001),
                  biases_init=Constant(0))
        mlp.initialize()
        probs = mlp.apply(x)
        #sq_err = SquaredError()
        cost = T.mean(T.sqr(y.flatten() - probs.flatten()))
        cost.name = 'cost'
        pred_out = probs > 0.5
        mis_cost = T.mean(T.neq(y.flatten(), pred_out.flatten()))
        mis_cost.name = 'MisclassificationRate'
        return mlp, cost, mis_cost

    def train_base_model(self, train_data, test_data, input_dim):
        x = T.matrix('features')
        y = T.matrix('targets')
        mlp, cost, mis_cost = self.create_base_model(x, y, input_dim)
        cg = ComputationGraph([cost])
        inputs = VariableFilter(roles=[INPUT])(cg.variables)
        cg = apply_dropout(cg, inputs, 0.2)
        algorithm = GradientDescent(cost=cost, parameters=cg.parameters,
                                    step_rule=Adam(learning_rate=0.001))
        data_stream = train_data
        data_stream_test = test_data
        monitor = DataStreamMonitoring(variables=[mis_cost],
                                       data_stream=data_stream_test,
                                       prefix="test")
        plot_ext = Plot('F1-measure',
                        channels=[['test_MisclassificationRate']],
                        after_batch=True)
        main_loop = MainLoop(data_stream=data_stream, algorithm=algorithm,
                             extensions=[monitor,
                                         FinishAfter(after_n_epochs=50),
                                         Printing(),
                                         plot_ext])
        main_loop.run()
        return mlp

    def calculate_measure(self, thresh,
                          y=None, y_hat=None, mlp=None,
                          beta=1):
        if y_hat is not None:
            pred = y_hat > thresh
            P = np.sum(y)
            N = np.sum(1 - y)
            TP = np.sum(pred * y * y_hat * (y_hat > thresh)) / P
            TN = np.sum((1 - pred) * (1 - y) * (1 - y_hat)
                        * (y_hat < thresh)) / N
            print (np.sum(pred * y), np.sum((1 - pred) * (1 - y)), P, N)
            f_measure = (1 + beta**2) * TP / (beta**2 +
                                              self.theta +
                                              TP - self.theta * TN)
            # f_measure = fbeta_score(
            #    y, pred, average='binary', pos_label=1, beta=beta)
            # cnf = confusion_matrix(y, pred)
            print TP, TN, thresh
            # print f_measure, f_measure_orig, "kk"
        return f_measure

    def find_new_thresh(self, value, beta=1,
                        y=None, y_hat=None, measure='f_beta'):

        fmeas_coef = {'a': (0, 1 + beta**2, 0),
                      'b': (beta**2 + self.theta, 1, -1 * self.theta)
                      }

        if measure == 'f_beta':
            coeff = fmeas_coef

        def get_value_coeff(coeff):
            c = coeff['a'][0] / coeff['b'][0]
            alpha = coeff['a'][1] / coeff['b'][0]
            beta = coeff['a'][2] / coeff['b'][0]
            gamma = coeff['b'][1] / coeff['b'][0]
            delta = coeff['b'][2] / coeff['b'][0]

            coeff1 = alpha - value * gamma / 2
            coeff2 = beta - value * delta / 2
            return c, coeff1, coeff2

        c, coeff1, coeff2 = get_value_coeff(coeff)
        assert(coeff1 >= 0), "Coefficient 1 is negative"
        assert(coeff2 >= 0), "Coefficient 2 is negative"
        normConst = coeff1 + coeff2
        print coeff1, coeff2, value, 'coeffs'
        thresh = coeff2 / normConst

        return thresh

    def altMax(self,  y, y_hat, tol=10e-10, thresh=0.0):
        nu = 0
        nu_arr = []
        thresh = thresh
        nu = self.calculate_measure(thresh, y=y, y_hat=y_hat)
        print "First Value is ", nu
        i = 0
        while True:
            new_thresh = nu / 2  # self.find_new_thresh(nu)
            nu_t = self.calculate_measure(new_thresh, y=y, y_hat=y_hat)
            print nu_t, new_thresh, "nu_t, thresh"
            if abs(nu - nu_t) < tol and i > 100:
                break
            else:
                i += 1
                nu = nu_t
                nu_arr.append(nu)
            thresh = new_thresh
        return (thresh, nu_arr)

    def get_prob(self, mlp, example_set, scheme, batch_size=256):
        dataset_state = example_set.open()
        x = T.matrix('x')
        out = mlp.apply(x)
        pred_fn = theano.function([x], out)
        y = np.zeros(example_set.num_examples)
        y_hat = np.zeros(example_set.num_examples)
        for idx, request in enumerate(scheme.get_request_iterator()):
            data = example_set.get_data(state=dataset_state, request=request)
            out_val = pred_fn(data[0])
            end_idx = (idx + 1) * batch_size
            if end_idx < example_set.num_examples:
                y[idx * batch_size:end_idx] = data[1].flatten()
                y_hat[idx * batch_size:end_idx] = out_val.flatten()

        return (y, y_hat)

if __name__ == '__main__':
    X = np.random.randn(10, 5)
    y = np.asarray([0.,  0.,  1.,  0.,  1.,  1.,  0.,  1.,  0.,  1.])
    sm = FbetaOpt()
    train_fn, test_fn = sm.get_fns(input_dim=5)
