from lightning.impl.dataset_fast cimport RowDataset

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
                    int id_transformer,
                    double[:, ::1] random_weights,
                    double[:] offset,
                    int[:] orders,
                    double[:] p_choice,
                    double[:] coefs_maclaurin,
                    double[:] z_cache,
                    int[:] hash_indices,
                    int[:] hash_signs,
                    int degree,
                    int kernel,
                    double[:] anova,
                    )
