#include "common_functions.cuh"

namespace generated_kernels {

using namespace generated_kernels::indexing;

// The wrapper function called by Fortran
extern "C" {
    void cpp_start_hot() {
    }

    void cpp_finish_hot() {
    }

    void cpp_CDU(
        double* __restrict__ u, size_t u_dim1, size_t u_dim2, size_t u_dim3,
        double* __restrict__ u2, size_t u2_dim1, size_t u2_dim2, size_t u2_dim3,
        int unx,
        int uny,
        int unz,
        double* __restrict__ v, size_t v_dim1, size_t v_dim2, size_t v_dim3,
        double* __restrict__ w, size_t w_dim1, size_t w_dim2, size_t w_dim3,
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
        double vadv;
        double wadv;
        double zero;

        zero = 0.0;
        half = 0.5;
        
        #pragma omp parallel for private(j, i)
        for (k = 2; k <= (unz + 1); k++) {
            for (j = 2; j <= (uny + 1); j++) {
                for (i = 2; i <= (unx + 1); i++) {
                    u2[F_IDX(i, j, k, u2_dim1, u2_dim2, u2_dim3)] = zero;
                }
            }
        }
        
        ax = (0.25 / dxmin);
        ay = (0.25 / dymin);
        az = (0.25 / dzmin);
        
        #pragma omp parallel for private(j, i)
        for (k = 2; k <= (unz + 1); k++) {
            for (j = 2; j <= (uny + 1); j++) {
                for (i = 2; i <= (unx + 1); i++) {
                    u2[F_IDX(i, j, k, u2_dim1, u2_dim2, u2_dim3)] = (-(((((((ax * ((u[F_IDX((i + 1), j, k, u_dim1, u_dim2, u_dim3)] + u[F_IDX(i, j, k, u_dim1, u_dim2, u_dim3)]))) * ((u[F_IDX((i + 1), j, k, u_dim1, u_dim2, u_dim3)] + u[F_IDX(i, j, k, u_dim1, u_dim2, u_dim3)]))) - ((ax * ((u[F_IDX(i, j, k, u_dim1, u_dim2, u_dim3)] + u[F_IDX((i - 1), j, k, u_dim1, u_dim2, u_dim3)]))) * ((u[F_IDX(i, j, k, u_dim1, u_dim2, u_dim3)] + u[F_IDX((i - 1), j, k, u_dim1, u_dim2, u_dim3)]))))) + ((((ay * ((u[F_IDX(i, (j + 1), k, u_dim1, u_dim2, u_dim3)] + u[F_IDX(i, j, k, u_dim1, u_dim2, u_dim3)]))) * ((v[F_IDX((i + 1), j, k, v_dim1, v_dim2, v_dim3)] + v[F_IDX(i, j, k, v_dim1, v_dim2, v_dim3)]))) - ((ay * ((u[F_IDX(i, j, k, u_dim1, u_dim2, u_dim3)] + u[F_IDX(i, (j - 1), k, u_dim1, u_dim2, u_dim3)]))) * ((v[F_IDX((i + 1), (j - 1), k, v_dim1, v_dim2, v_dim3)] + v[F_IDX(i, (j - 1), k, v_dim1, v_dim2, v_dim3)])))))) + ((((az * ((u[F_IDX(i, j, (k + 1), u_dim1, u_dim2, u_dim3)] + u[F_IDX(i, j, k, u_dim1, u_dim2, u_dim3)]))) * ((w[F_IDX((i + 1), j, k, w_dim1, w_dim2, w_dim3)] + w[F_IDX(i, j, k, w_dim1, w_dim2, w_dim3)]))) - ((az * ((u[F_IDX(i, j, k, u_dim1, u_dim2, u_dim3)] + u[F_IDX(i, j, (k - 1), u_dim1, u_dim2, u_dim3)]))) * ((w[F_IDX((i + 1), j, (k - 1), w_dim1, w_dim2, w_dim3)] + w[F_IDX(i, j, (k - 1), w_dim1, w_dim2, w_dim3)]))))))));
                }
            }
        }
        
        ax = (0.5 / dxmin);
        ay = (0.125 / dymin);
        az = (0.125 / dzmin);
        
        #pragma omp parallel for private(vadv, wadv, j, i)
        for (k = 2; k <= (unz + 1); k++) {
            for (j = 2; j <= (uny + 1); j++) {
                for (i = 2; i <= (unx + 1); i++) {
                    vadv = ((((v[F_IDX(i, j, k, v_dim1, v_dim2, v_dim3)] + v[F_IDX((i + 1), j, k, v_dim1, v_dim2, v_dim3)]) + v[F_IDX(i, (j - 1), k, v_dim1, v_dim2, v_dim3)]) + v[F_IDX((i + 1), (j - 1), k, v_dim1, v_dim2, v_dim3)]));
                    wadv = ((((w[F_IDX(i, j, k, w_dim1, w_dim2, w_dim3)] + w[F_IDX((i + 1), j, k, w_dim1, w_dim2, w_dim3)]) + w[F_IDX(i, j, (k - 1), w_dim1, w_dim2, w_dim3)]) + w[F_IDX((i + 1), j, (k - 1), w_dim1, w_dim2, w_dim3)]));
                    u2[F_IDX(i, j, k, u2_dim1, u2_dim2, u2_dim3)] = (u2[F_IDX(i, j, k, u2_dim1, u2_dim2, u2_dim3)] - (((((ax * ((u[F_IDX((i + 1), j, k, u_dim1, u_dim2, u_dim3)] - u[F_IDX((i - 1), j, k, u_dim1, u_dim2, u_dim3)]))) * u[F_IDX(i, j, k, u_dim1, u_dim2, u_dim3)]) + ((ay * ((u[F_IDX(i, (j + 1), k, u_dim1, u_dim2, u_dim3)] - u[F_IDX(i, (j - 1), k, u_dim1, u_dim2, u_dim3)]))) * vadv)) + ((az * ((u[F_IDX(i, j, (k + 1), u_dim1, u_dim2, u_dim3)] - u[F_IDX(i, j, (k - 1), u_dim1, u_dim2, u_dim3)]))) * wadv))));
                }
            }
        }
        
        #pragma omp parallel for private(j, i)
        for (k = 2; k <= (unz + 1); k++) {
            for (j = 2; j <= (uny + 1); j++) {
                for (i = 2; i <= (unx + 1); i++) {
                    u2[F_IDX(i, j, k, u2_dim1, u2_dim2, u2_dim3)] = (u2[F_IDX(i, j, k, u2_dim1, u2_dim2, u2_dim3)] * half);
                }
            }
        }
        
    }
}
}