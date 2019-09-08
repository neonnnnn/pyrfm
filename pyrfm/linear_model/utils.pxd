from ..random_feature.random_mapping cimport BaseCRandomFeature


cdef void normalize(double[:] z,
                    double[:] mean,
                    double[:] var,
                    int t,
                    Py_ssize_t n_components)


cdef void transform(X_array,
                    double[:] z,
                    Py_ssize_t i,
                    double* data,
                    int* indices,
                    int n_nz,
                    bint is_sparse,
                    transformer,
                    BaseCRandomFeature transformer_fast
                    )