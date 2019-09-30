import numpy as np
from .sparse_rademacher import sparse_rademacher


def next_pow_of_two(x):
    if not isinstance(x, int):
        raise TypeError("x is not integer.")
    if x < 0:
        raise ValueError("x is negative.")
    return 1 if x == 0 else 2**(x-1).bit_length()


# 0 mean and 1 variance distribution
def standard_gaussian(random_state, size):
    return random_state.randn(*size)


def rademacher(random_state, size):
    return random_state.randint(2, size=size, dtype=np.int32)*2-1


def laplace(random_state, size):
    return random_state.laplace(0, 1. / np.sqrt(2), size)


def uniform(random_state, size):
    return random_state.uniform(-np.sqrt(3), np.sqrt(3), size)


def get_random_matrix(random_state, distribution, size, p_sparse=0.,
                      dtype=np.float64):
    # size = (n_components, n_features)
    if distribution == 'rademacher':
        return rademacher(random_state, size).astype(dtype)
    elif distribution in ['gaussian', 'normal']:
        return standard_gaussian(random_state, size)
    elif distribution == 'uniform':
        return uniform(random_state, size)
    elif distribution == 'laplace':
        return laplace(random_state, size)
    elif distribution == 'sparse_rademacher':
        # n_nzs : (n_features, )
        # n_nzs[j] is n_nz of random_weights[:, j]
        return sparse_rademacher(random_state, np.array(size, dtype=np.int32),
                                 p_sparse)
    else:
        raise ValueError('{} distribution is not implemented. Please use'
                         'rademacher, gaussian (normal), uniform or laplace.'
                         .format(distribution))
