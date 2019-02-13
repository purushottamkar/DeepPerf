from DAMP.FMeasure import FbetaOpt
from datasets.dataRead import a9aReader
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
from fuel.schemes import ShuffledScheme
# Create the objects

# Spade Optimizer
# model = Spade(dual_class)
# Data reader
a9a = a9aReader(location='./datasets/kdd08.')
a9a.read()
input_dim = a9a.input_dim
print "The input dimension is " + str(input_dim)
num_splits = 0
batch_size = 1024
train_dataset, test_dataset, p = a9a.get_split(0)
print "Number of training examples is " + str(train_dataset.num_examples)
print "Number of testing examples is " + str(test_dataset.num_examples)

model = FbetaOpt(p)

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
num_epochs = 10

cost = []
time_arr = []

passed_time = 0
curr_time = time.time()
idx = 0
for epochs in tqdm(xrange(num_epochs)):
    for request in scheme.get_request_iterator():
        data = train_dataset.get_data(state=train_state, request=request)
        c = train_fn(data[0], data[1].ravel())
        #print c
        if idx % 100 == 0:
            time_arr.append(passed_time + time.time() - curr_time)
            passed_time = time_arr[-1]
            test_req = range(test_dataset.num_examples)
            test_data = test_dataset.get_data(state=test_state,
                                              request=test_req)
            minC = test_fn(test_data[0], test_data[1])
            cost.append(minC)
            curr_time = time.time()
        idx += 1
train_dataset.close(train_state)
test_dataset.close(test_state)
print cost, time_arr
plt.plot(time_arr, cost)
plt.show()
