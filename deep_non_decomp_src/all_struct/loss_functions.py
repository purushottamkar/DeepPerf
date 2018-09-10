import numpy as np
import math
loss_dict = {
    'miss_classfn': 1,
    'prec': 2,
    'rec': 3,
    'hmean': 4,
    'qmean': 5,
    'fone': 6,
    'kldnorm': 7,
    'ber': 8,
    'nas': 9,
    'nss': 10,
    'mTPRTNR': 11
}


def get_loss_fn(loss_code):
    if loss_code == 1:
        return miss_classfn
    elif loss_code == 2:
        return prec
    elif loss_code == 3:
        return rec
    elif loss_code == 4:
        return hmean
    elif loss_code == 5:
        return qmean
    elif loss_code == 6:
        return fmeasure
    elif loss_code == 7:
        return kldnorm
    elif loss_code == 11:
        return minTPRTNR


def miss_classfn(y, y_pred):
    return np.mean(y == y_pred)


def minTPRTNR(y, y_pred):
    TPR = np.sum((y == 1) * (y_pred == 1)) * 1.0 / np.sum(y == 1)
    TNR = np.sum((y == -1) * (y_pred == -1)) * 1.0 / np.sum(y == -1)
    print TPR, TNR
    return min(TPR, TNR)


def prec(y, y_pred):
    return (np.sum((y == 1) * (y_pred == 1)) + 10e-7) \
        / (10e-7 + np.sum((y_pred == 1)))


def rec(y, y_pred):

    return np.sum((y == 1) * (y_pred == 1)) * 1.0 / np.sum(y == 1)


def hmean(y, y_pred):

    TNR = np.sum((1 - y) * (1 - y_pred))
    TPR = np.sum(y * y_pred) / np.sum(y)
    return 1 - 2 * TPR * TNR / (TPR + TNR)


def qmean(y, y_pred):
    TPR = np.sum((y == 1) * (y_pred == 1)) * 1.0 / np.sum(y == 1)
    TNR = np.sum((y == -1) * (y_pred == -1)) * 1.0 / np.sum(y == -1)
    print "TPR : " + str(TPR) + " TNR " + str(TNR)
    print y_pred[0:10], y[0:10]
    return 1 - math.sqrt(((1 - TPR)**2 + (1 - TNR)**2) / 2.0)


def fmeasure(y, y_pred):
    precV = prec(y, y_pred)
    print "PREC" + str(precV)
    recV = rec(y, y_pred)
    print "REC" + str(recV)
    return (2 * precV * recV / (precV + recV))


def kldnorm(y, y_pred):
    eps = 1.0 / (2.0 * y.shape[0])
    p = np.sum(y == 1) * 1.0 / y.shape[0]
    phat = np.sum(y_pred == 1) * 1.0 / y.shape[0]
    preg = regularize(p, eps)
    phatreg = regularize(phat, eps)
    print p, phat, preg, phatreg, "IN"
    loss = -(preg * 1.0 * np.log(preg / phatreg) +
             (1 - preg) * 1.0 *
             np.log((1 - preg) * 1.0 / (1.0 - phatreg))) / np.log(1 / eps)
    return loss


def regularize(p, eps):
    return (p + eps) * 1.0 / (1 + 2.0 * eps)
