# cython: language_level=3
# cython: cdivision=True
# cython: boundscheck=False
# cython: wraparound=False
from libc.math cimport  sqrt
import numpy as np
cimport numpy as np
from ..random_feature.random_features_fast cimport BaseCRandomFeature
from ..random_feature.random_features_doubly cimport BaseCDoublyRandomFeature
from cython.view cimport array
from ..dataset_fast cimport RowDataset


cdef void normalize(double[:] z,
                    double[:] mean,
                    double[:] var,
                    int t,
                    Py_ssize_t n_components):
    cdef double mean_new
    cdef Py_ssize_t j
    for j in range(n_components):
        mean_new = mean[j] + (z[j] - mean[j]) / (t+1)
        var[j] = var[j] * (1-1./t)
        var[j] += (z[j] - mean[j])*(z[j] - mean_new) / t
        mean[j] = mean_new
        z[j] = (z[j] - mean[j]) / (1e-6 + sqrt(var[j]))


cdef void transform(X_array,
                    double[:] z,
                    Py_ssize_t i,
                    double* data,
                    int* indices,
                    int n_nz,
                    bint is_sparse,
                    transformer,
                    BaseCRandomFeature transformer_fast
                    ):
    cdef Py_ssize_t j, n_components
    if transformer_fast is None:
        if is_sparse:
            _z = transformer.transform(X_array[i])[0]
        else:
            _z = transformer.transform(np.atleast_2d(X_array[i]))[0]
        n_components = z.shape[0]
        for j in range(n_components):
            z[j] = _z[j]
    else:
        transformer_fast.transform(&z[0], data, indices, n_nz)


def _predict_fast_doubly(double[:] coef,
                         RowDataset X,
                         double[:] y_pred,
                         BaseCDoublyRandomFeature transformer_fast):
    cdef Py_ssize_t n_samples, n_components, i
    n_samples = X.get_n_samples()
    n_components = coef.shape[0]
    cdef int[:] indices_samples = array((n_samples, ), sizeof(int), 
                                        format='i')
    for i in range(n_samples):
        indices_samples[i] = i
    transformer_fast.pred_batch(None, y_pred, coef, 0, X,
                                &indices_samples[0], n_samples,
                                n_components, -1)

 
def _predict_fast(double[:] coef,
                  RowDataset X,
                  double[:] y_pred,
                  double[:] mean,
                  double[:] var,
                  BaseCRandomFeature transformer_fast):
    cdef Py_ssize_t n_samples, n_components, j, i
    cdef double[:] z
    # data pointers
    cdef int* indices
    cdef double* data
    cdef int n_nz
    n_samples = X.get_n_samples()
    n_components = transformer_fast.get_n_components()
    z = array((n_components, ), sizeof(double), format='d')
    for j in range(n_components):
        z[j] = 0
    for i in range(n_samples):
        X.get_row_ptr(i, &indices, &data, &n_nz)
        transformer_fast.transform(&z[0], data, indices, n_nz)
        # if normalize
        if mean is not None:
            for j in range(n_components):
                z[j] = (z[j] - mean[j]) / (1e-6 + sqrt(var[j]))

        y_pred[i] = 0
        for j in range(n_components):
            y_pred[i] += z[j] * coef[j]
