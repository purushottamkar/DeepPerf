from DAMP.AMP import FbetaThresh
from datasets.dataRead import a9aReader
import matplotlib.pyplot as plt
from tqdm import tqdm
import time
import sys
from fuel.schemes import ShuffledScheme
from fuel.streams import DataStream
import numpy as np
# Create the objects

# Spade Optimizer
# model = Spade(dual_class)
# Data reader
fName = 'a9a'
if len(sys.argv) > 1:
    fName = sys.argv[1]
a9a = a9aReader(location='./datasets/' + fName + '.')
a9a.read()
input_dim = a9a.input_dim
print "The input dimension is " + str(input_dim)
num_splits = 0
batch_size = 1024
train_dataset, test_dataset, p = a9a.get_split(0)
print "Number of training examples is " + str(train_dataset.num_examples)
print "Number of testing examples is " + str(test_dataset.num_examples)

model = FbetaThresh(p)

train_state = train_dataset.open()
test_state = test_dataset.open()

scheme = ShuffledScheme(examples=train_dataset.num_examples,
                        batch_size=batch_size)
test_scheme = ShuffledScheme(examples=test_dataset.num_examples,
                             batch_size=batch_size)
print "Input dim is " + str(input_dim)

print "Basic training beginning"
print "p is " + str(p)
# Get the theano functions
data_stream = DataStream(dataset=train_dataset, iteration_scheme=scheme)
test_data_stream = DataStream(dataset=test_dataset,
                              iteration_scheme=test_scheme)
mlp = model.train_base_model(data_stream, test_data_stream, input_dim)
(y, y_hat) = model.get_prob(mlp, train_dataset, scheme, batch_size=batch_size)
thresh, nu = model.altMax(y, y_hat, thresh=0.5)

print thresh, nu
plt.plot(nu)
plt.show()
np.savez(open(fName + 'AMP_PG.npz', 'wb+'), nu)
