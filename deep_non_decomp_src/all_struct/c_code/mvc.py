import ctypes as ct
import numpy as np
import os


def mvc(y, scores, p, n, loss_fn):
    #y = 2 * y - 1
    _doublepp = ct.POINTER(ct.c_double)
    _mvc = ct.CDLL(os.getcwd() + '/all_struct/c_code/libmvc.so')
    _mvc.getMostViolatedConstraint.argtypes = (_doublepp,
                                               _doublepp,
                                               ct.c_int,
                                               ct.c_int,
                                               ct.c_int,
                                               ct.c_int,
                                               _doublepp)
    y = y.astype(np.float64)
    scores = scores.astype(np.float64)
    yhat = np.zeros_like(y).astype(np.float64)
    _mvc.getMostViolatedConstraint(
        y.ctypes.data_as(ct.POINTER(ct.c_double)),
        scores.ctypes.data_as(ct.POINTER(ct.c_double)),
        ct.c_int(int(loss_fn)),
        ct.c_int(int(p + n)),
        ct.c_int(int(p)),
        ct.c_int(int(n)),
        yhat.ctypes.data_as(ct.POINTER(ct.c_double)),

    )
    #print p, n
    #print yhat[0:100], y[0:100], scores[0:100]
    return np.float32(yhat) - np.float32(y)

if __name__ == '__main__':
    print mvc(np.asarray([1, 1, 1, -1, -1]),
              np.asarray([1.0, -2, 0.9, -0.9, -9.9]),
              3,
              2,
              4)
