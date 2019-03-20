from fuel.datasets import IndexableDataset
from collections import OrderedDict

import scipy.io
import numpy as np


class a9aReader(object):

    def __init__(self, location='./datasets/kdd08.'):
        print "Dataset : " + location
        self.x_data_location = location + 'X.mat'
        self.y_data_location = location + 'y.mat'
        self.splits_data_location = location + 'splits.mat'

    def read(self, ):
        data_load = scipy.io.loadmat(self.x_data_location)['X']
        if scipy.sparse.issparse(data_load):
            self.X_data = data_load.toarray().T
        else:
            self.X_data = data_load.T
        self.input_dim = self.X_data.shape[1]

        self.y_data = np.asarray(scipy.io.loadmat(
            self.y_data_location)['y'].ravel())
        splits_data = scipy.io.loadmat(self.splits_data_location)
        splits = splits_data['splits']
        self.num_valid = splits['numValidating'].ravel()[0][0][0]
        self.idxs = np.asarray(splits['IDs'][0][0]) - 1

    def get_numpy_split(self, split_num=0):
        split_perm = self.idxs[:, split_num]
        train_idx = split_perm[:-1 * self.num_valid]
        test_idx = split_perm[-1 * self.num_valid:]
        X_train = self.X_data[train_idx].astype(np.float32)
        y_train = self.y_data[train_idx].astype(np.float32)
        X_test = self.X_data[test_idx].astype(np.float32)
        y_test = self.y_data[test_idx].astype(np.float32)

        def convertbin(y_temp):
            y_temp += 1
            y_temp /= 2
            return y_temp
        y_train = convertbin(y_train)
        y_test = convertbin(y_test)

        p = np.sum(y_train) * 1.0 / (X_train.shape[0])
        return X_train, y_train, X_test, y_test, np.float32(p)

    def get_split(self, split_num=0):
        split_perm = self.idxs[:, split_num]
        train_idx = split_perm[:-self.num_valid]
        test_idx = split_perm[-1 * self.num_valid:]
        X_train = self.X_data[train_idx]
        y_train = self.y_data[train_idx]
        X_test = self.X_data[test_idx]
        y_test = self.y_data[test_idx]

        def convertbin(y_temp):
            y_temp += 1
            y_temp /= 2
            return y_temp
        y_train = convertbin(y_train).reshape(-1, 1)
        y_test = convertbin(y_test).reshape(-1, 1)
        train_dataset = IndexableDataset(indexables=OrderedDict(
            [('features', X_train.astype(np.float32)),
             ('targets', y_train.astype(np.float32))]))

        test_dataset = IndexableDataset(indexables=OrderedDict(
            [('features', X_test.astype(np.float32)),
             ('targets', y_test.astype(np.float32))]))

        p = np.sum(y_train) * 1.0 / (X_train.shape[0])
        return train_dataset, test_dataset, np.float32(p)

if __name__ == '__main__':
    a9aR = a9aReader()
    a9aR.read()
    X_train, y_train, X_test, y_test = a9aR.get_split(0)
