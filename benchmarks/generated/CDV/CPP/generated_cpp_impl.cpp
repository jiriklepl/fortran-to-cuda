#include "common_functions.cuh"

namespace generated_kernels {

using namespace generated_kernels::indexing;

// The wrapper function called by Fortran
extern "C" {
    void cpp_start_hot() {
    }

    void cpp_finish_hot() {
    }

    void cpp_CDV(
        double* __restrict__ u, size_t u_dim1, size_t u_dim2, size_t u_dim3,
        double* __restrict__ v, size_t v_dim1, size_t v_dim2, size_t v_dim3,
        double* __restrict__ v2, size_t v2_dim1, size_t v2_dim2, size_t v2_dim3,
        int vnx,
        int vny,
        int vnz,
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
        double uadv;
        double wadv;
        double zero;

        zero = 0.0;
        half = 0.5;
        
        for (k = 2; k <= (vnz + 1); k++) {
            for (j = 2; j <= (vny + 1); j++) {
                for (i = 2; i <= (vnx + 1); i++) {
                    v2[F_IDX(i, j, k, v2_dim1, v2_dim2, v2_dim3)] = zero;
                }
            }
        }
        
        ax = (0.25 / dxmin);
        ay = (0.25 / dymin);
        az = (0.25 / dzmin);
        
        for (k = 2; k <= (vnz + 1); k++) {
            for (j = 2; j <= (vny + 1); j++) {
                for (i = 2; i <= (vnx + 1); i++) {
                    v2[F_IDX(i, j, k, v2_dim1, v2_dim2, v2_dim3)] = (-(((((((ay * ((v[F_IDX(i, (j + 1), k, v_dim1, v_dim2, v_dim3)] + v[F_IDX(i, j, k, v_dim1, v_dim2, v_dim3)]))) * ((v[F_IDX(i, (j + 1), k, v_dim1, v_dim2, v_dim3)] + v[F_IDX(i, j, k, v_dim1, v_dim2, v_dim3)]))) - ((ay * ((v[F_IDX(i, j, k, v_dim1, v_dim2, v_dim3)] + v[F_IDX(i, (j - 1), k, v_dim1, v_dim2, v_dim3)]))) * ((v[F_IDX(i, j, k, v_dim1, v_dim2, v_dim3)] + v[F_IDX(i, (j - 1), k, v_dim1, v_dim2, v_dim3)]))))) + ((((ax * ((v[F_IDX((i + 1), j, k, v_dim1, v_dim2, v_dim3)] + v[F_IDX(i, j, k, v_dim1, v_dim2, v_dim3)]))) * ((u[F_IDX(i, (j + 1), k, u_dim1, u_dim2, u_dim3)] + u[F_IDX(i, j, k, u_dim1, u_dim2, u_dim3)]))) - ((ax * ((v[F_IDX(i, j, k, v_dim1, v_dim2, v_dim3)] + v[F_IDX((i - 1), j, k, v_dim1, v_dim2, v_dim3)]))) * ((u[F_IDX((i - 1), (j + 1), k, u_dim1, u_dim2, u_dim3)] + u[F_IDX((i - 1), j, k, u_dim1, u_dim2, u_dim3)])))))) + ((((az * ((v[F_IDX(i, j, (k + 1), v_dim1, v_dim2, v_dim3)] + v[F_IDX(i, j, k, v_dim1, v_dim2, v_dim3)]))) * ((w[F_IDX(i, (j + 1), k, w_dim1, w_dim2, w_dim3)] + w[F_IDX(i, j, k, w_dim1, w_dim2, w_dim3)]))) - ((az * ((v[F_IDX(i, j, k, v_dim1, v_dim2, v_dim3)] + v[F_IDX(i, j, (k - 1), v_dim1, v_dim2, v_dim3)]))) * ((w[F_IDX(i, (j + 1), (k - 1), w_dim1, w_dim2, w_dim3)] + w[F_IDX(i, j, (k - 1), w_dim1, w_dim2, w_dim3)]))))))));
                }
            }
        }
        
        ax = (0.125 / dxmin);
        ay = (0.5 / dymin);
        az = (0.125 / dzmin);
        
        for (k = 2; k <= (vnz + 1); k++) {
            for (j = 2; j <= (vny + 1); j++) {
                for (i = 2; i <= (vnx + 1); i++) {
                    uadv = ((((u[F_IDX(i, j, k, u_dim1, u_dim2, u_dim3)] + u[F_IDX(i, (j + 1), k, u_dim1, u_dim2, u_dim3)]) + u[F_IDX((i - 1), j, k, u_dim1, u_dim2, u_dim3)]) + u[F_IDX((i - 1), (j + 1), k, u_dim1, u_dim2, u_dim3)]));
                    wadv = ((((w[F_IDX(i, j, k, w_dim1, w_dim2, w_dim3)] + w[F_IDX(i, (j + 1), k, w_dim1, w_dim2, w_dim3)]) + w[F_IDX(i, j, (k - 1), w_dim1, w_dim2, w_dim3)]) + w[F_IDX(i, (j + 1), (k - 1), w_dim1, w_dim2, w_dim3)]));
                    v2[F_IDX(i, j, k, v2_dim1, v2_dim2, v2_dim3)] = (v2[F_IDX(i, j, k, v2_dim1, v2_dim2, v2_dim3)] - (((((ax * ((v[F_IDX((i + 1), j, k, v_dim1, v_dim2, v_dim3)] - v[F_IDX((i - 1), j, k, v_dim1, v_dim2, v_dim3)]))) * uadv) + ((ay * ((v[F_IDX(i, (j + 1), k, v_dim1, v_dim2, v_dim3)] - v[F_IDX(i, (j - 1), k, v_dim1, v_dim2, v_dim3)]))) * v[F_IDX(i, j, k, v_dim1, v_dim2, v_dim3)])) + ((az * ((v[F_IDX(i, j, (k + 1), v_dim1, v_dim2, v_dim3)] - v[F_IDX(i, j, (k - 1), v_dim1, v_dim2, v_dim3)]))) * wadv))));
                }
            }
        }
        
        for (k = 2; k <= (vnz + 1); k++) {
            for (j = 2; j <= (vny + 1); j++) {
                for (i = 2; i <= (vnx + 1); i++) {
                    v2[F_IDX(i, j, k, v2_dim1, v2_dim2, v2_dim3)] = (v2[F_IDX(i, j, k, v2_dim1, v2_dim2, v2_dim3)] * half);
                }
            }
        }
        
    }
}
}
