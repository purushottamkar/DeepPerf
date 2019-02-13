from DeeSpade.batch_model import Spade
from DeeSpade.bench import BenchANN
from datasets.dataRead import a9aReader
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import sys
from DeeSpade.dual_step import QMean, MinTPRTNR
from fuel.schemes import ShuffledScheme
import numpy as np
# Create the objects

# Dual Updates (Choose from MinTPRTNR, QMean)
dual_class = MinTPRTNR()


# Spade Optimizer or Bench Optimizaer
# model = Spade(dual_class)
model = BenchANN(dual_class)

# Dataset from argument
location = 'a9a'
if len(sys.argv) > 1:
    location = sys.argv[1]
# Data reader
a9a = a9aReader(location='./datasets/' + location + '.')
a9a.read()
input_dim = a9a.input_dim
print "The input dimension is " + str(input_dim)
num_splits = 0
batch_size = 256
train_dataset, test_dataset, p = a9a.get_split(0)
print "Number of training examples is " + str(train_dataset.num_examples)
print "Number of testing examples is " + str(test_dataset.num_examples)
train_state = train_dataset.open()
test_state = test_dataset.open()

scheme = ShuffledScheme(examples=train_dataset.num_examples,
                        batch_size=batch_size)
test_scheme = ShuffledScheme(examples=test_dataset.num_examples,
                             batch_size=test_dataset.num_examples)
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
minTPRTNR = []
qmean = []
f_meas = []
kld = []
ba = []
flag = False
for epochs in tqdm(xrange(num_epochs)):
    for request in scheme.get_request_iterator():
        data = train_dataset.get_data(state=train_state, request=request)
        train_fn(data[0], data[1].ravel())

        if idx % 100 == 0:
            time_arr.append(passed_time + time.time() - curr_time)
            test_req = range(test_dataset.num_examples)
            passed_time = time_arr[-1]
            test_data = test_dataset.get_data(state=test_state,
                                              request=test_req)
            minC, out = test_fn(test_data[0], test_data[1])
            # qmean.append(minC)
            minTPRTNR.append(minC)

            # When using Bench ANN code uncomment following
            # minTPRTNR.append(minC[0])
            # qmean.append(minC[1])
            # f_meas.append(minC[2])
            # kld.append(minC[3])
            # ba.append(minC[4])

            curr_time = time.time()
        idx += 1
    if flag:
        break

test_req = range(test_dataset.num_examples)
test_data = test_dataset.get_data(state=test_state,
                                  request=test_req)
minC, out = test_fn(test_data[0], test_data[1])
np.savetxt(location + '_bench.csv', out)
train_dataset.close(train_state)
test_dataset.close(test_state)
print cost, time_arr

#np.savez(open(location + 'SPADE_Qmean_TC.npz', 'wb+'), qmean)
#np.savez(open(location + 'SPADE_MinTPRTNR.npz', 'wb+'), minTPRTNR)
np.savez(open(location + 'pk_bench.npz', 'wb+'), minTPRTNR)
#np.savez(open(location + 'ANN_MinTPRTNR.npz', 'wb+'), minTPRTNR)
#np.savez(open(location + 'ANN_fmeas.npz', 'wb+'), f_meas)
#np.savez(open(location + 'ANN_kld.npz', 'wb+'), kld)
#np.savez(open(location + 'ANN_ba.npz', 'wb+'), ba)
# plt.plot(f_meas)
plt.plot(minTPRTNR)
# plt.plot(qmean)
# plt.plot(kld)
# plt.plot(ba)
#plt.legend(['F_meas', 'minTPRTNR', 'qmean', 'kld', 'ba'])
plt.show()
