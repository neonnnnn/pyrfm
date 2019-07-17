import numpy as np


def generate_samples(n_samples, n_features, seed=0):
    random_state = np.random.RandomState(seed)
    size = (n_samples, n_features)
    X = random_state.random_sample(size=size) * 2 - 1
    return X


def generate_target(X_trans, rng, low=-1.0, high=1.0):
    coef = rng.uniform(low, high, size=X_trans.shape[1])
    y = np.dot(X_trans, coef)
    y -= np.mean(y)
    return y, coef
