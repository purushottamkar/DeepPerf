
from demesis.denembis import DeeNemBis
from datasets.dataRead import a9aReader
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import sys
from demesis.concave_fn import KLD
from fuel.schemes import ShuffledScheme
import numpy as np
#from utils import twin_plot
np.set_printoptions(threshold=np.nan)
# Create the objects

# Dual Updates

location = 'a9a'
if len(sys.argv) > 1:
    location = sys.argv[1]
    # Data reader
a9a = a9aReader(location='./datasets/' + location + '.')
a9a.read()
input_dim = a9a.input_dim
print "The input dimension is " + str(input_dim)
batch_size = 512
train_dataset, test_dataset, p = a9a.get_split(0)
print "Number of training examples is " + str(train_dataset.num_examples)
print "Number of testing examples is " + str(test_dataset.num_examples)


'''__author__: Amartya Sanyal <amartya18x@gmail.com>'''


def get_results(C=None, pt=None):
    dual_class = KLD(C)

    # Direct Otimizer
    # model = CCmodel()

    # Spade Optimizer
    model = DeeNemBis(dual_class)
    train_state = train_dataset.open()
    test_state = test_dataset.open()

    scheme = ShuffledScheme(examples=train_dataset.num_examples,
                            batch_size=batch_size)
    # test_scheme = ShuffledScheme(examples=test_dataset.num_examples,
    #                             batch_size=test_dataset.num_examples)
    print "Input dim is " + str(input_dim)
    print "p is " + str(p)
    # Get the theano functions
    train_fn, test_fn = model.get_fns(input_dim=input_dim, p=p)
    num_epochs = 100

    cost = []
    time_arr = []

    passed_time = 0
    curr_time = time.time()
    idx = 0
    meas1 = []
    meas2 = []
    for epochs in tqdm(xrange(num_epochs)):
        for request in scheme.get_request_iterator():
            data = train_dataset.get_data(
                state=train_state, request=request)
            r0, r1 = train_fn(data[0], data[1].ravel())
            # print r0, r1
            if idx % 100 == 0:
                 # print a, b
                time_arr.append(passed_time + time.time() - curr_time)
                passed_time = time_arr[-1]
                test_req = range(test_dataset.num_examples)
                test_data = test_dataset.get_data(state=test_state,
                                                  request=test_req)
                minC, out = test_fn(test_data[0], test_data[1])

                meas1.append(minC)
                curr_time = time.time()
            idx += 1
    minC, out = test_fn(test_data[0], test_data[1])
    np.savetxt('gold_' + location + '.csv', test_data[1])
    np.savetxt('test_' + location + '.csv', out)
    train_dataset.close(train_state)
    test_dataset.close(test_state)
    # print meas1, meas2
    return time_arr, meas1


def eval_KLD():
    curr_time, cost = get_results(p)
    print len(curr_time), len(cost), "KKKKK"
    filename = location + '_kld_rew.npz'
    fileP = open(filename, 'wb+')
    np.savez(fileP, curr_time, cost)
    fileP.close()
    n = 3
    cost = np.convolve(cost, np.ones(n) / n)[n - 1:1 - n]
    plt.plot(cost)
    plt.show()


def eval_BAKLD():
    meas1 = []
    meas2 = []
    cost = []
    C_vary = np.exp(-np.arange(10, 1, -0.3)) * 3
    for x in C_vary:
        a, b, c = get_results(x)
        meas1.append(a)
        meas2.append(b)
        cost.append(c)
    file_name = location + '_meas'
    fileP = open(file_name, 'wb+')
    np.savez(fileP, meas1, meas2, cost)
    #twin_plt = twin_plot(C_vary, meas1, meas2, 'BA', '-KLD')
    #twin_plt.savefig(location + '_C' + str(x) + '_twin.png')
    #plt.plot(C_vary, cost)
    # plt.show()
    #plt.savefig(location + '_cost.png')


if __name__ == '__main__':
    eval_KLD()
