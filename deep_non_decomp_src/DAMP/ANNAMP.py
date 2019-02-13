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
from fuel.datasets import IndexableDataset
from fuel.streams import DataStream
from fuel.schemes import ShuffledScheme
from collections import OrderedDict

import theano.tensor as T
import theano
import numpy as np

theano.config.optimizer = 'None'
theano.config.exception_verbosity = 'high'


class FbetaANN(object):

    def __init__(self, p, beta=1, num_epoch=10):
        self.p = p
        self.num_epoch = num_epoch
        self.beta = beta
        self.p = p
        self.theta = (1 - p) * 1.0 / p

    def create_base_model(self, x, y, input_dim, interim_dim=30):

        # Create the output of the MLP
        mlp = MLP([Tanh(), Tanh(),  Tanh()],
                  [input_dim, 60, 60, interim_dim],
                  weights_init=IsotropicGaussian(0.001),
                  biases_init=Constant(0))
        mlp.initialize()
        inter = mlp.apply(x)

        fine_tuner = MLP([Logistic()],
                         [interim_dim, 1],
                         weights_init=IsotropicGaussian(0.001),
                         biases_init=Constant(0))
        fine_tuner.initialize()
        probs = fine_tuner.apply(inter)
        #sq_err = BinaryCrossEntropy()
        err = T.sqr(y.flatten() - probs.flatten())
        # cost = T.mean(err * y.flatten() * (1 - self.p) + err *
        #              (1 - y.flatten()) * self.p)
        cost = T.mean(err)
        #cost = sq_err.apply(probs.flatten(), y.flatten())
        # cost = T.mean(y.flatten() * T.log(probs.flatten()) +
        #              (1 - y.flatten()) * T.log(1 - probs.flatten()))
        cost.name = 'cost'
        pred_out = probs > 0.5
        mis_cost = T.sum(T.neq(y.flatten(), pred_out.flatten()))
        mis_cost.name = 'MisclassificationRate'
        return mlp, fine_tuner, cost, mis_cost

    def train_base_model(self, train_data, test_data, input_dim, interim_dim):
        x = T.matrix('features')
        y = T.matrix('targets')
        mlp, fine_tuner, cost, mis_cost = self.create_base_model(
            x, y, input_dim, interim_dim=interim_dim)
        cg = ComputationGraph([cost])
        inputs = VariableFilter(roles=[INPUT])(cg.variables)
        cg = apply_dropout(cg, inputs, 0.6)
        algorithm = GradientDescent(cost=cost, parameters=cg.parameters,
                                    step_rule=Adam(learning_rate=0.01))
        data_stream = train_data
        data_stream_test = test_data
        monitor = DataStreamMonitoring(variables=[mis_cost],
                                       data_stream=data_stream_test,
                                       prefix="test")
        # plot_ext = Plot('F1-measure',
        #                channels=[['test_MisclassificationRate']],
        #                after_batch=True)
        main_loop = MainLoop(data_stream=data_stream, algorithm=algorithm,
                             extensions=[monitor,
                                         FinishAfter(after_n_epochs=30),
                                         Printing(),
                                         #                                         plot_ext
                                         ])
        main_loop.run()
        return (mlp, fine_tuner)

    def calculate_measure(self,  example_set, mlp=None, thresh=0.5, beta=1):

        (y, y_hat), P, N = self.get_curr_out(mlp, example_set)
        if y_hat is not None:
            pred = y_hat > thresh
            TP = np.sum(pred * y * y_hat) / P
            TN = np.sum((1 - pred) * (1 - y) * (1 - y_hat)) / N
            print (np.sum(pred * y * y_hat),
                   np.sum((1 - pred) * (1 - y) * (1 - y_hat)), P, N)
            f_measure = (1 + beta**2) * TP / (beta**2 +
                                              self.theta +
                                              TP - self.theta * TN)
            # f_measure = fbeta_score(
            #    y, pred, average='binary', pos_label=1, beta=beta)
            # cnf = confusion_matrix(y, pred)
            print np.sum(pred), np.sum(y_hat), TN, thresh, "Look", P, N,
            # print f_measure, f_measure_orig, "kk"
        P = np.sum(y)
        N = np.sum(1 - y)

        return f_measure, P, N

    def get_fine_tuner_fn(self, value, fine_tuner, dataset, P, N, beta=1,
                          batch_size=256,
                          y=None, y_hat=None, measure='f_beta',
                          fine_tune_epoch=30):

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

            coeff1 = alpha - value * gamma
            coeff2 = beta - value * delta
            return c, coeff1, coeff2

        c, coeff1, coeff2 = get_value_coeff(coeff)
        assert(coeff1 >= 0), "Coefficient 1 is negative"
        assert(coeff2 >= 0), "Coefficient 2 is negative"

        x = T.matrix('features')
        y = T.matrix('targets')
        probs = fine_tuner.apply(x).flatten()
        y = y.flatten()
        #P = T.sum(y)
        #N = T.sum(1 - y)

        pred_pos = probs > 0.5
        pred_neg = probs <= 0.5

        TPR = T.sum(probs * y * pred_pos) / P
        TNR = T.sum((1 - probs) * (1 - y) * pred_neg) / N
        print coeff1, coeff2, "Hellow"
        cost = (-1 * coeff1) * TPR + (-1 * coeff2) * TNR
        #cost = T.sum(probs) + T.sum(y)
        cost.name = "cost"

        cg = ComputationGraph([cost])
        algorithm = GradientDescent(cost=cost, parameters=cg.parameters,
                                    step_rule=Adam(learning_rate=0.001))
        scheme = ShuffledScheme(examples=dataset.num_examples,
                                batch_size=batch_size)
        datastream = DataStream(dataset=dataset, iteration_scheme=scheme)
        main_loop = MainLoop(data_stream=datastream, algorithm=algorithm,
                             extensions=[
                                 FinishAfter(after_n_epochs=fine_tune_epoch)])
        main_loop.run()
        return fine_tuner

    def altMax(self,  model, dataset, tol=10e-10, thresh=0.0,
               fine_tune_epoch=30, amp_round=30, test_dataset=None):
        if test_dataset is None:
            test_dataset = dataset

        (mlp, fine_tuner) = model
        nu = 0
        nu_arr = []
        thresh = thresh
        i = 0
        value, P, N = self.calculate_measure(dataset, fine_tuner)
        print "First Value is ", value
        while True:
            fine_tuner = self.get_fine_tuner_fn(
                value=value, fine_tuner=fine_tuner, dataset=dataset,
                P=P, N=N, fine_tune_epoch=fine_tune_epoch)
            nu_t, P, N = self.calculate_measure(test_dataset, fine_tuner)
            print nu_t, "nu_t"
            if i == amp_round:
                break
            else:
                i += 1
                nu = nu_t
                nu_arr.append(nu)
        return nu_arr

    def get_curr_out(self, model, example_set, batch_size=256):
        scheme = ShuffledScheme(examples=example_set.num_examples,
                                batch_size=batch_size)
        example_state = example_set.open()
        x = T.matrix('features')
        out = model.apply(x)
        pred_fn = theano.function([x], out)
        y = np.zeros((example_set.num_examples))
        y_hat = np.zeros((example_set.num_examples))
        for idx, request in enumerate(scheme.get_request_iterator()):
            data = example_set.get_data(state=example_state, request=request)
            out_val = pred_fn(data[0].astype(np.float32))
            end_idx = (idx + 1) * batch_size
            if end_idx < example_set.num_examples:
                y[idx * batch_size:end_idx] = data[1].flatten()
                y_hat[idx * batch_size:end_idx] = out_val.flatten()
        P = np.sum(y)
        N = np.sum(1 - y)

        return (y, y_hat), P, N

    def get_prob(self, model, example_set, scheme, interim_dim=30,
                 batch_size=256):
        (mlp, fine_tuner) = model
        dataset_state = example_set.open()
        x = T.matrix('x')
        out = mlp.apply(x)
        pred_fn = theano.function([x], out)
        y = np.zeros((example_set.num_examples))
        print "Number of examples is ", example_set.num_examples
        y_hat = np.zeros((example_set.num_examples, interim_dim))
        for idx, request in enumerate(scheme.get_request_iterator()):
            data = example_set.get_data(state=dataset_state, request=request)
            out_val = pred_fn(data[0])
            end_idx = (idx + 1) * batch_size
            if end_idx < example_set.num_examples:
                y[idx * batch_size:end_idx] = data[1].flatten()
                y_hat[idx * batch_size:end_idx] = out_val
        dataset = IndexableDataset(indexables=OrderedDict(
            [('features', y_hat.astype(np.float32)),
             ('targets', y.reshape(-1, 1).astype(np.float32))]))
        return dataset


if __name__ == '__main__':
    X = np.random.randn(10, 5)
    y = np.asarray([0.,  0.,  1.,  0.,  1.,  1.,  0.,  1.,  0.,  1.])
    sm = FbetaOpt()
    train_fn, test_fn = sm.get_fns(input_dim=5)
