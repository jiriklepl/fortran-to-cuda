#include "common_functions.cuh"

namespace generated_kernels {

using namespace generated_kernels::indexing;

// The wrapper function called by Fortran
extern "C" {
    void cpp_start_hot() {
    }

    void cpp_finish_hot() {
    }

    void cpp_CDW(
        double* __restrict__ u, size_t u_dim1, size_t u_dim2, size_t u_dim3,
        double* __restrict__ v, size_t v_dim1, size_t v_dim2, size_t v_dim3,
        double* __restrict__ w, size_t w_dim1, size_t w_dim2, size_t w_dim3,
        double* __restrict__ w2, size_t w2_dim1, size_t w2_dim2, size_t w2_dim3,
        int wnx,
        int wny,
        int wnz,
        double dxmin,
        double dymin,
        double dzmin
    ) {
        double ax;
        double ay;
        double az;
        double half;
        int i;
        int j;
        int k;
        double uadv;
        double vadv;
        double zero;

        zero = 0.0;
        half = 0.5;
        
        #pragma omp parallel for private(j, i)
        for (k = 2; k <= (wnz + 1); k++) {
            for (j = 2; j <= (wny + 1); j++) {
                for (i = 2; i <= (wnx + 1); i++) {
                    w2[F_IDX(i, j, k, w2_dim1, w2_dim2, w2_dim3)] = zero;
                }
            }
        }
        
        ax = (0.25 / dxmin);
        ay = (0.25 / dymin);
        az = (0.25 / dzmin);
        
        #pragma omp parallel for private(j, i)
        for (k = 2; k <= (wnz + 1); k++) {
            for (j = 2; j <= (wny + 1); j++) {
                for (i = 2; i <= (wnx + 1); i++) {
                    w2[F_IDX(i, j, k, w2_dim1, w2_dim2, w2_dim3)] = (-(((((((az * ((w[F_IDX(i, j, (k + 1), w_dim1, w_dim2, w_dim3)] + w[F_IDX(i, j, k, w_dim1, w_dim2, w_dim3)]))) * ((w[F_IDX(i, j, (k + 1), w_dim1, w_dim2, w_dim3)] + w[F_IDX(i, j, k, w_dim1, w_dim2, w_dim3)]))) - ((az * ((w[F_IDX(i, j, k, w_dim1, w_dim2, w_dim3)] + w[F_IDX(i, j, (k - 1), w_dim1, w_dim2, w_dim3)]))) * ((w[F_IDX(i, j, k, w_dim1, w_dim2, w_dim3)] + w[F_IDX(i, j, (k - 1), w_dim1, w_dim2, w_dim3)]))))) + ((((ay * ((w[F_IDX(i, (j + 1), k, w_dim1, w_dim2, w_dim3)] + w[F_IDX(i, j, k, w_dim1, w_dim2, w_dim3)]))) * ((v[F_IDX(i, j, (k + 1), v_dim1, v_dim2, v_dim3)] + v[F_IDX(i, j, k, v_dim1, v_dim2, v_dim3)]))) - ((ay * ((w[F_IDX(i, j, k, w_dim1, w_dim2, w_dim3)] + w[F_IDX(i, (j - 1), k, w_dim1, w_dim2, w_dim3)]))) * ((v[F_IDX(i, (j - 1), k, v_dim1, v_dim2, v_dim3)] + v[F_IDX(i, (j - 1), (k + 1), v_dim1, v_dim2, v_dim3)])))))) + ((((ax * ((w[F_IDX((i + 1), j, k, w_dim1, w_dim2, w_dim3)] + w[F_IDX(i, j, k, w_dim1, w_dim2, w_dim3)]))) * ((u[F_IDX(i, j, (k + 1), u_dim1, u_dim2, u_dim3)] + u[F_IDX(i, j, k, u_dim1, u_dim2, u_dim3)]))) - ((ax * ((w[F_IDX(i, j, k, w_dim1, w_dim2, w_dim3)] + w[F_IDX((i - 1), j, k, w_dim1, w_dim2, w_dim3)]))) * ((u[F_IDX((i - 1), j, (k + 1), u_dim1, u_dim2, u_dim3)] + u[F_IDX((i - 1), j, k, u_dim1, u_dim2, u_dim3)]))))))));
                }
            }
        }
        
        ax = (0.125 / dxmin);
        ay = (0.125 / dymin);
        az = (0.5 / dzmin);
        
        #pragma omp parallel for private(uadv, vadv, j, i)
        for (k = 2; k <= (wnz + 1); k++) {
            for (j = 2; j <= (wny + 1); j++) {
                for (i = 2; i <= (wnx + 1); i++) {
                    uadv = ((((u[F_IDX(i, j, k, u_dim1, u_dim2, u_dim3)] + u[F_IDX(i, j, (k + 1), u_dim1, u_dim2, u_dim3)]) + u[F_IDX((i - 1), j, k, u_dim1, u_dim2, u_dim3)]) + u[F_IDX((i - 1), j, (k + 1), u_dim1, u_dim2, u_dim3)]));
                    vadv = ((((v[F_IDX(i, j, k, v_dim1, v_dim2, v_dim3)] + v[F_IDX(i, j, (k + 1), v_dim1, v_dim2, v_dim3)]) + v[F_IDX(i, (j - 1), k, v_dim1, v_dim2, v_dim3)]) + v[F_IDX(i, (j - 1), (k + 1), v_dim1, v_dim2, v_dim3)]));
                    w2[F_IDX(i, j, k, w2_dim1, w2_dim2, w2_dim3)] = (w2[F_IDX(i, j, k, w2_dim1, w2_dim2, w2_dim3)] - (((((ax * ((w[F_IDX((i + 1), j, k, w_dim1, w_dim2, w_dim3)] - w[F_IDX((i - 1), j, k, w_dim1, w_dim2, w_dim3)]))) * uadv) + ((ay * ((w[F_IDX(i, (j + 1), k, w_dim1, w_dim2, w_dim3)] - w[F_IDX(i, (j - 1), k, w_dim1, w_dim2, w_dim3)]))) * vadv)) + ((az * ((w[F_IDX(i, j, (k + 1), w_dim1, w_dim2, w_dim3)] - w[F_IDX(i, j, (k - 1), w_dim1, w_dim2, w_dim3)]))) * w[F_IDX(i, j, k, w_dim1, w_dim2, w_dim3)]))));
                }
            }
        }
        
        #pragma omp parallel for private(j, i)
        for (k = 2; k <= (wnz + 1); k++) {
            for (j = 2; j <= (wny + 1); j++) {
                for (i = 2; i <= (wnx + 1); i++) {
                    w2[F_IDX(i, j, k, w2_dim1, w2_dim2, w2_dim3)] = (w2[F_IDX(i, j, k, w2_dim1, w2_dim2, w2_dim3)] * half);
                }
            }
        }
        
    }
}
}