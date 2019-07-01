cdef void random_fourier(double[:] z,
                         double* data,
                         int* indices,
                         int n_nz,
                         double[:, ::1] random_weights,
                         double[:] bias
                         )

cdef void random_kernel(double[:] z,
                        double* data,
                        int* indices,
                        int n_nz,
                        double[:, ::1] random_weights,
                        int kernel,
                        int degree,
                        double[:] a)


cdef void random_maclaurin(double[:] z,
                           double* data,
                           int* indices,
                           int n_nz,
                           double[:, ::1] random_weights,
                           int[:] orders,
                           double[:] p_choice,
                           double[:] coefs,
                           )

cdef void tensor_sketch(double[:] z,
                        double[:] z_cache,
                        double* data,
                        int* indices,
                        int n_nz,
                        int degree,
                        int[:] hash_indices,
                        int[:] hash_signs)