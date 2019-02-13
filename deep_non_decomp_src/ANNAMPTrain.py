from DAMP.ANNAMP import FbetaANN
from datasets.dataRead import a9aReader
import matplotlib.pyplot as plt
from fuel.schemes import ShuffledScheme
from fuel.streams import DataStream
import sys
import os
import numpy as np
# Create the objects

# Spade Optimizer
# Data reader
dataset = 'ppi'
if len(sys.argv) > 1:
    dataset = sys.argv[1]
try:
    a9a = a9aReader(location='./datasets/' + dataset + '.')
except:
    print "Wrong dataset"
    os.exit()
a9a.read()
input_dim = a9a.input_dim
print "The input dimension is " + str(input_dim)
num_splits = 0
batch_size = 512
interim_dim = 80
fine_tuner_epoch = 3
amp_round = 100
train_dataset, test_dataset, p = a9a.get_split(0)
print "Number of training examples is " + str(train_dataset.num_examples)
print "Number of testing examples is " + str(test_dataset.num_examples)

model = FbetaANN(p)

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
mlp = model.train_base_model(data_stream, test_data_stream,
                             input_dim, interim_dim)
dataset_f = model.get_prob(mlp, train_dataset, scheme,
                           batch_size=batch_size,
                           interim_dim=interim_dim)
dataset_ftest = model.get_prob(mlp, test_dataset, test_scheme,
                               batch_size=batch_size,
                               interim_dim=interim_dim)
nu = model.altMax(mlp, dataset_f, scheme, thresh=0.5,
                  fine_tune_epoch=fine_tuner_epoch, amp_round=amp_round,
                  test_dataset=dataset_f)

print nu
# plt.plot(nu)
# plt.show()

np.savez(open(dataset + 'ANNAMAP_FMeas_new.npz', 'wb+'), nu)
