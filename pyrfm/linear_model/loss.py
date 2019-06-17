import numpy as np


def sigmoid(pred):
    return np.exp(np.minimum(0, pred)) / (1.+np.exp(-np.abs(pred)))


class LossFunction:
    def loss(self, p, y):
        raise NotImplementedError()

    def dloss(self, p, y):
        raise NotImplementedError()


class Squared(LossFunction):
    """Squared loss: L(p, y) = 0.5 * (y - p)²"""

    def __init__(self):
        self.mu = 1

    def loss(self, p, y):
        return 0.5 * (p - y) ** 2

    def dloss(self, p, y):
        return p - y


class Logistic(LossFunction):
    """Logistic loss: L(p, y) = log(1 + exp(-yp))"""

    def __init__(self):
        self.mu = 0.25

    def loss(self, p, y):
        z = p * y
        # log(1 + exp(-z))

        return np.log(1.0 + np.exp(-np.maximum(0, z))) - np.minimum(0, z)

    def dloss(self, p, y):
        z = p * y
        # def tau = 1 / (1 + exp(-z))
        # return y * (tau - 1)
        tau = sigmoid(z)
        return y * (tau - 1)


class SquaredHinge(LossFunction):
    """Squared hinge loss: L(p, y) = max(1 - yp, 0)²"""

    def __init__(self):
        self.mu = 2

    def loss(self, p, y):
        z = 1 - p * y
        if z > 0:
            return z * z
        else:
            return 0.0

    def dloss(self, p, y):
        z = 1 - p * y
        z[z < 0] = 0
        z[z > 0] *= -2*y[z > 0]
        return z


class Hinge(LossFunction):
    """hinge loss: L(p, y) = max(1 - y*p, 0)"""

    def __init__(self):
        self.mu = 0

    def loss(self, p, y):
        z = 1 - p * y
        z[z < 0] = 0
        return z

    def dloss(self, p, y):
        z = 1 - p * y
        z[z < 0] = 0
        z[z > 0] = -1
        return z