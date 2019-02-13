from all_struct.struct_ann import struct_ann
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import sys
from datasets.dataRead import a9aReader
from fuel.schemes import ShuffledScheme
import numpy as np
# Create the objects

model = struct_ann()
location = 'a9a'
if len(sys.argv) > 1:
    location = sys.argv[1]
    loss_fn = sys.argv[2]

# Data reader
a9a = a9aReader(location='./datasets/' + location + '.')
a9a.read()
input_dim = a9a.input_dim
print "The input dimension is " + str(input_dim)
num_splits = 0
batch_size = 6000
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
train_fn, test_fn = model.get_fns(input_dim=input_dim, loss=loss_fn)
num_epochs = 40

cost = []
time_arr = []

passed_time = 0
curr_time = time.time()
idx = 0
score = []
for epochs in tqdm(xrange(num_epochs)):
    for request in scheme.get_request_iterator():
        data = train_dataset.get_data(state=train_state, request=request)
        train_fn(data[0], data[1].ravel())

        if idx % 1 == 0:
            # print a, b
            time_arr.append(passed_time + time.time() - curr_time)
            passed_time = time_arr[-1]
            test_req = range(test_dataset.num_examples)
            test_data = test_dataset.get_data(state=test_state,
                                              request=test_req)
            loss = test_fn(test_data[0], test_data[1])
            score.append(loss)
            curr_time = time.time()
        idx += 1


train_dataset.close(train_state)
test_dataset.close(test_state)
print cost, time_arr

np.savez(open(location + 'struct_'+loss_fn+'_.npz', 'wb+'), score)
plt.plot(score)
plt.show()
