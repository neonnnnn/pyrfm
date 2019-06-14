import numpy as np
from scipy.fftpack import fft, ifft


def singed_circulant_random_projection(X, fft_random_weights, signs):
    # compute the approximation of the product of random matrix and vector.
    n_samples, n_features = X.shape
    t = fft_random_weights.shape[0]
    n_components = n_features * t

    if fft_random_weights.shape[1] != n_features:
        raise ValueError('random_weights.shape must be (t, n_features)'
                         '(t * n_features = n_components).')

    if signs.shape != fft_random_weights.shape:
        raise ValueError('shape of random_weights and signs must be same.')

    random_features = np.zeros((n_samples, n_components))
    fft_X = fft(X)
    for t, fft_rw, s in enumerate(zip(fft_random_weights, signs)):
        projection = s*ifft(fft_X*fft_rw).real
        random_features[:, t*n_features:(t+1)*n_features] = projection

    return random_features
