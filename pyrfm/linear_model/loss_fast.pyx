# cython: language_level=3
# cython: cdivision=True

from libc.math cimport log, exp, fmax, fmin, fabs

cdef class LossFunction:

    cdef double loss(self, double p, double y):
        raise NotImplementedError()

    cdef double dloss(self, double p, double y):
        raise NotImplementedError()

    cdef double conjugate(self, double alpha, double y):
        raise NotImplementedError()

    cdef double sdca_update(self, double alpha, double y, double p,
                            double scale):
        raise NotImplementedError()


cdef class Squared(LossFunction):
    """Squared loss: L(p, y) = 0.5 * (y - p)²"""

    def __init__(self):
        self.mu = 1

    cdef double loss(self, double p, double y):
        return 0.5 * (p - y) ** 2

    cdef double dloss(self, double p, double y):
        return p - y

    cdef double conjugate(self, double alpha, double y):
        return alpha*y + 0.5*alpha**2

    cdef double sdca_update(self, double alpha, double y, double p,
                            double scale):
        return (y-alpha-p) / (1+scale)


cdef class Logistic(LossFunction):
    """Logistic loss: L(p, y) = log(1 + exp(-yp))"""

    def __init__(self):
        self.mu = 0.25

    cdef double loss(self, double p, double y):
        cdef double z = p * y
        # log(1 + exp(-z))
        if z > 18:
            return exp(-z)
        elif z < -18:
            return -z
        else:
            return log(1.0 + exp(-z))

    cdef double dloss(self, double p, double y):
        cdef double z = p * y
        # cdef double tau = 1 / (1 + exp(-z))
        # return y * (tau - 1)
        if z > 18.0:
            return -y * exp(-z)
        elif z < -18.0:
            return -y
        else:
            return -y / (exp(z) + 1.0)

    cdef double conjugate(self, double alpha, double y):
        cdef double z = alpha*y
        if z == 0:
            return 0
        elif z == -1:
            return 0
        elif (-z < 1) and (-z > 0):
            return -z*log(-z) + (1+z) * log(1+z)
        else:
            raise ValueError("alpha*y is bigger than 0 or lower than -1")

    cdef double sdca_update(self, double alpha, double y, double p,
                            double scale):
        cdef double update = y*(exp(fmin(0, -p*y))/(1+exp(-fabs(p*y))))-alpha
        return update / fmax(1, 0.25+scale)


cdef class SquaredHinge(LossFunction):
    """Squared hinge loss: L(p, y) = max(1 - yp, 0)²"""

    def __init__(self):
        self.mu = 2

    cdef double loss(self, double p, double y):
        cdef double z = 1 - p * y
        if z > 0:
            return z * z
        else:
            return 0.0

    cdef double dloss(self, double p, double y):
        cdef double z = 1 - p * y
        if z > 0:
            return -2 * y * z
        else:
            return 0.0

    cdef double conjugate(self, double alpha, double y):
        if alpha * y > 0:
            raise ValueError("alpha*y > 0")
        else:
            return alpha*y + alpha*alpha/4

    cdef double sdca_update(self, double alpha, double y, double p,
                            double scale):
        cdef double update = (y-p-0.5*alpha) / (0.5+scale)
        if (alpha + update) * y < 0:
            return -alpha
        else:
            return update

cdef class Hinge(LossFunction):
    """hinge loss: L(p, y) = max(1 - y*p, 0)"""

    def __init__(self):
        self.mu = 0

    cdef double loss(self, double p, double y):
        cdef double z = 1 - p * y
        if z > 0:
            return z
        else:
            return 0.0

    cdef double dloss(self, double p, double y):
        cdef double z = 1 - p * y
        if z > 0:
            return -1
        else:
            return 0.0

    cdef double conjugate(self, double alpha, double y):
        return alpha*y

    cdef double sdca_update(self, double alpha, double y, double p,
                            double scale):
        cdef double z = (1-y*p) / scale + alpha*y
        return y*fmax(0, fmin(1, z)) - alpha

