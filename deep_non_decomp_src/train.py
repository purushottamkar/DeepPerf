from DeeSpade.model import Spade
from datasets.dataRead import a9aReader
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import numpy as np
import sys
# Create the objects

# Spade Model
sm = Spade()

# Data reader
filename = 'a9a'

if len(sys.argv) > 1:
    filename = sys.argv[1]

a9a = a9aReader(location='./datasets/'+filename+'.')
a9a.read()
input_dim = a9a.input_dim
num_splits = 0
X_train, y_train, X_test, y_test, p = a9a.get_numpy_split(0)

print "Input dim is " + str(input_dim)
print "p is " + str(p)
# Get the theano functions
train_fn, test_fn = sm.get_fns(input_dim=input_dim, p=p)
num_epochs = 10

cost = []
time_arr = []
y_train = np.asarray(y_train, np.float32)
y_test = np.asarray(y_test, np.float32)

passed_time = 0
curr_time = time.time()
for epochs in tqdm(xrange(num_epochs)):
    for idx in xrange(X_train.shape[0]):
        train_fn(X_train[idx], y_train[idx])
        if idx % 100 == 0:
            time_arr.append(passed_time + time.time() - curr_time)
            passed_time = time_arr[-1]
            minC, out = test_fn(X_test, y_test)
            cost.append(minC)
            curr_time = time.time()

print cost, time_arr
plt.plot(time_arr, cost)
plt.show()
