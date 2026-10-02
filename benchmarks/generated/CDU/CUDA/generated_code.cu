#include <cuda_runtime.h>
#include <cstddef>
#include <iostream>

#ifdef USE_PINNED_MEMORY
#include <unordered_set>
#endif

#define MEASURE_CUDA_EXECUTION_TIME
#include "common_functions.cuh"

using namespace generated_kernels::indexing;
using namespace generated_kernels::timing;

#ifdef USE_PINNED_MEMORY
std::unordered_set<void*> pinned_ptrs;
#endif

namespace generated_kernels {

__global__ 
void kernel_group_1_device(
    int unz,
    int unx,
    double zero,
    double* __restrict__ u2, size_t u2_dim1, size_t u2_dim2, size_t u2_dim3,
    int uny,
    int k_from, int k_to,
    int j_from, int j_to,
    int i_from, int i_to,
    size_t total_elements
) {
    // 1D Thread Index
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Ensure we don't go out of bounds
    if (idx >= total_elements) {
        return;
    }

    // Declarations of local variables used in the kernel body
    int i;
    int j;
    int k;

    // Map the 1D index back to column-major multi-dimensional coordinates
    constexpr int k_step = 1;
        constexpr int j_step = 1;
        constexpr int i_step = 1;
    size_t current_idx = idx;
    
    // Calculate index for 'i' dimension
    int num_i = ((i_to - i_from + i_step) / i_step);
    int local_i = current_idx % num_i;
    i = i_from + local_i * i_step;
    current_idx /= num_i;
    
    // Calculate index for 'j' dimension
    int num_j = ((j_to - j_from + j_step) / j_step);
    int local_j = current_idx % num_j;
    j = j_from + local_j * j_step;
    current_idx /= num_j;
    
    // Calculate index for 'k' dimension
    int num_k = ((k_to - k_from + k_step) / k_step);
    int local_k = current_idx;
    k = k_from + local_k * k_step;

    // Perform the calculation
    u2[F_IDX(i, j, k, u2_dim1, u2_dim2, u2_dim3)] = zero;
    
}

__global__ 
void kernel_group_3_device(
    int unz,
    double az,
    double* __restrict__ w, size_t w_dim1, size_t w_dim2, size_t w_dim3,
    double ax,
    int unx,
    double* __restrict__ v, size_t v_dim1, size_t v_dim2, size_t v_dim3,
    double* __restrict__ u, size_t u_dim1, size_t u_dim2, size_t u_dim3,
    double* __restrict__ u2, size_t u2_dim1, size_t u2_dim2, size_t u2_dim3,
    double ay,
    int uny,
    int k_from, int k_to,
    int j_from, int j_to,
    int i_from, int i_to,
    size_t total_elements
) {
    // 1D Thread Index
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Ensure we don't go out of bounds
    if (idx >= total_elements) {
        return;
    }

    // Declarations of local variables used in the kernel body
    int i;
    int j;
    int k;

    // Map the 1D index back to column-major multi-dimensional coordinates
    constexpr int k_step = 1;
        constexpr int j_step = 1;
        constexpr int i_step = 1;
    size_t current_idx = idx;
    
    // Calculate index for 'i' dimension
    int num_i = ((i_to - i_from + i_step) / i_step);
    int local_i = current_idx % num_i;
    i = i_from + local_i * i_step;
    current_idx /= num_i;
    
    // Calculate index for 'j' dimension
    int num_j = ((j_to - j_from + j_step) / j_step);
    int local_j = current_idx % num_j;
    j = j_from + local_j * j_step;
    current_idx /= num_j;
    
    // Calculate index for 'k' dimension
    int num_k = ((k_to - k_from + k_step) / k_step);
    int local_k = current_idx;
    k = k_from + local_k * k_step;

    // Perform the calculation
    u2[F_IDX(i, j, k, u2_dim1, u2_dim2, u2_dim3)] = (-(((((((ax * ((u[F_IDX((i + 1), j, k, u_dim1, u_dim2, u_dim3)] + u[F_IDX(i, j, k, u_dim1, u_dim2, u_dim3)]))) * ((u[F_IDX((i + 1), j, k, u_dim1, u_dim2, u_dim3)] + u[F_IDX(i, j, k, u_dim1, u_dim2, u_dim3)]))) - ((ax * ((u[F_IDX(i, j, k, u_dim1, u_dim2, u_dim3)] + u[F_IDX((i - 1), j, k, u_dim1, u_dim2, u_dim3)]))) * ((u[F_IDX(i, j, k, u_dim1, u_dim2, u_dim3)] + u[F_IDX((i - 1), j, k, u_dim1, u_dim2, u_dim3)]))))) + ((((ay * ((u[F_IDX(i, (j + 1), k, u_dim1, u_dim2, u_dim3)] + u[F_IDX(i, j, k, u_dim1, u_dim2, u_dim3)]))) * ((v[F_IDX((i + 1), j, k, v_dim1, v_dim2, v_dim3)] + v[F_IDX(i, j, k, v_dim1, v_dim2, v_dim3)]))) - ((ay * ((u[F_IDX(i, j, k, u_dim1, u_dim2, u_dim3)] + u[F_IDX(i, (j - 1), k, u_dim1, u_dim2, u_dim3)]))) * ((v[F_IDX((i + 1), (j - 1), k, v_dim1, v_dim2, v_dim3)] + v[F_IDX(i, (j - 1), k, v_dim1, v_dim2, v_dim3)])))))) + ((((az * ((u[F_IDX(i, j, (k + 1), u_dim1, u_dim2, u_dim3)] + u[F_IDX(i, j, k, u_dim1, u_dim2, u_dim3)]))) * ((w[F_IDX((i + 1), j, k, w_dim1, w_dim2, w_dim3)] + w[F_IDX(i, j, k, w_dim1, w_dim2, w_dim3)]))) - ((az * ((u[F_IDX(i, j, k, u_dim1, u_dim2, u_dim3)] + u[F_IDX(i, j, (k - 1), u_dim1, u_dim2, u_dim3)]))) * ((w[F_IDX((i + 1), j, (k - 1), w_dim1, w_dim2, w_dim3)] + w[F_IDX(i, j, (k - 1), w_dim1, w_dim2, w_dim3)]))))))));
    
}

__global__ 
void kernel_group_5_device(
    double az,
    int unz,
    double ax,
    double* __restrict__ w, size_t w_dim1, size_t w_dim2, size_t w_dim3,
    int unx,
    double* __restrict__ v, size_t v_dim1, size_t v_dim2, size_t v_dim3,
    double* __restrict__ u, size_t u_dim1, size_t u_dim2, size_t u_dim3,
    double ay,
    double* __restrict__ u2, size_t u2_dim1, size_t u2_dim2, size_t u2_dim3,
    int uny,
    int k_from, int k_to,
    int j_from, int j_to,
    int i_from, int i_to,
    size_t total_elements
) {
    // 1D Thread Index
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Ensure we don't go out of bounds
    if (idx >= total_elements) {
        return;
    }

    // Declarations of local variables used in the kernel body
    int i;
    int j;
    int k;
    double vadv;
    double wadv;

    // Map the 1D index back to column-major multi-dimensional coordinates
    constexpr int k_step = 1;
        constexpr int j_step = 1;
        constexpr int i_step = 1;
    size_t current_idx = idx;
    
    // Calculate index for 'i' dimension
    int num_i = ((i_to - i_from + i_step) / i_step);
    int local_i = current_idx % num_i;
    i = i_from + local_i * i_step;
    current_idx /= num_i;
    
    // Calculate index for 'j' dimension
    int num_j = ((j_to - j_from + j_step) / j_step);
    int local_j = current_idx % num_j;
    j = j_from + local_j * j_step;
    current_idx /= num_j;
    
    // Calculate index for 'k' dimension
    int num_k = ((k_to - k_from + k_step) / k_step);
    int local_k = current_idx;
    k = k_from + local_k * k_step;

    // Perform the calculation
    vadv = ((((v[F_IDX(i, j, k, v_dim1, v_dim2, v_dim3)] + v[F_IDX((i + 1), j, k, v_dim1, v_dim2, v_dim3)]) + v[F_IDX(i, (j - 1), k, v_dim1, v_dim2, v_dim3)]) + v[F_IDX((i + 1), (j - 1), k, v_dim1, v_dim2, v_dim3)]));
    wadv = ((((w[F_IDX(i, j, k, w_dim1, w_dim2, w_dim3)] + w[F_IDX((i + 1), j, k, w_dim1, w_dim2, w_dim3)]) + w[F_IDX(i, j, (k - 1), w_dim1, w_dim2, w_dim3)]) + w[F_IDX((i + 1), j, (k - 1), w_dim1, w_dim2, w_dim3)]));
    u2[F_IDX(i, j, k, u2_dim1, u2_dim2, u2_dim3)] = (u2[F_IDX(i, j, k, u2_dim1, u2_dim2, u2_dim3)] - (((((ax * ((u[F_IDX((i + 1), j, k, u_dim1, u_dim2, u_dim3)] - u[F_IDX((i - 1), j, k, u_dim1, u_dim2, u_dim3)]))) * u[F_IDX(i, j, k, u_dim1, u_dim2, u_dim3)]) + ((ay * ((u[F_IDX(i, (j + 1), k, u_dim1, u_dim2, u_dim3)] - u[F_IDX(i, (j - 1), k, u_dim1, u_dim2, u_dim3)]))) * vadv)) + ((az * ((u[F_IDX(i, j, (k + 1), u_dim1, u_dim2, u_dim3)] - u[F_IDX(i, j, (k - 1), u_dim1, u_dim2, u_dim3)]))) * wadv))));
    
}

__global__ 
void kernel_group_6_device(
    double half,
    int unz,
    int unx,
    double* __restrict__ u2, size_t u2_dim1, size_t u2_dim2, size_t u2_dim3,
    int uny,
    int k_from, int k_to,
    int j_from, int j_to,
    int i_from, int i_to,
    size_t total_elements
) {
    // 1D Thread Index
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Ensure we don't go out of bounds
    if (idx >= total_elements) {
        return;
    }

    // Declarations of local variables used in the kernel body
    int i;
    int j;
    int k;

    // Map the 1D index back to column-major multi-dimensional coordinates
    constexpr int k_step = 1;
        constexpr int j_step = 1;
        constexpr int i_step = 1;
    size_t current_idx = idx;
    
    // Calculate index for 'i' dimension
    int num_i = ((i_to - i_from + i_step) / i_step);
    int local_i = current_idx % num_i;
    i = i_from + local_i * i_step;
    current_idx /= num_i;
    
    // Calculate index for 'j' dimension
    int num_j = ((j_to - j_from + j_step) / j_step);
    int local_j = current_idx % num_j;
    j = j_from + local_j * j_step;
    current_idx /= num_j;
    
    // Calculate index for 'k' dimension
    int num_k = ((k_to - k_from + k_step) / k_step);
    int local_k = current_idx;
    k = k_from + local_k * k_step;

    // Perform the calculation
    u2[F_IDX(i, j, k, u2_dim1, u2_dim2, u2_dim3)] = (u2[F_IDX(i, j, k, u2_dim1, u2_dim2, u2_dim3)] * half);
    
}


// The wrapper function called by Fortran
extern "C" {
    void cpp_start_hot() {
        reset_timing_vectors();
    }

    void cpp_finish_hot() {
        print_timing_summary();
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
        #ifdef USE_PINNED_MEMORY

        if (pinned_ptrs.find(w) == pinned_ptrs.end()) {
            CUCH(cudaHostRegister(w, (sizeof(double) * w_dim1 * w_dim2 * w_dim3), cudaHostRegisterPortable));
            pinned_ptrs.insert(w);
        }
        if (pinned_ptrs.find(u2) == pinned_ptrs.end()) {
            CUCH(cudaHostRegister(u2, (sizeof(double) * u2_dim1 * u2_dim2 * u2_dim3), cudaHostRegisterPortable));
            pinned_ptrs.insert(u2);
        }
        if (pinned_ptrs.find(u) == pinned_ptrs.end()) {
            CUCH(cudaHostRegister(u, (sizeof(double) * u_dim1 * u_dim2 * u_dim3), cudaHostRegisterPortable));
            pinned_ptrs.insert(u);
        }
        if (pinned_ptrs.find(v) == pinned_ptrs.end()) {
            CUCH(cudaHostRegister(v, (sizeof(double) * v_dim1 * v_dim2 * v_dim3), cudaHostRegisterPortable));
            pinned_ptrs.insert(v);
        }

        #endif // USE_PINNED_MEMORY

        // 1. Allocate memory on the GPU (Device)
        double* w_device;
        double* u2_device;
        double* u_device;
        double* v_device;

        measure_alloc([&]() {
        CUCH(cudaMalloc(&w_device, (sizeof(double) * w_dim1 * w_dim2 * w_dim3)));
        CUCH(cudaMalloc(&u2_device, (sizeof(double) * u2_dim1 * u2_dim2 * u2_dim3)));
        CUCH(cudaMalloc(&u_device, (sizeof(double) * u_dim1 * u_dim2 * u_dim3)));
        CUCH(cudaMalloc(&v_device, (sizeof(double) * v_dim1 * v_dim2 * v_dim3)));
        });

        size_t total_h2d_bytes = (sizeof(double) * w_dim1 * w_dim2 * w_dim3) + (sizeof(double) * u2_dim1 * u2_dim2 * u2_dim3) + (sizeof(double) * u_dim1 * u_dim2 * u_dim3) + (sizeof(double) * v_dim1 * v_dim2 * v_dim3);

        // 2. Copy inputs from Host (CPU) to Device (GPU)
        measure_h2d(total_h2d_bytes, [&]() {
        CUCH(cudaMemcpy(w_device, w, (sizeof(double) * w_dim1 * w_dim2 * w_dim3), cudaMemcpyHostToDevice));
        CUCH(cudaMemcpy(u_device, u, (sizeof(double) * u_dim1 * u_dim2 * u_dim3), cudaMemcpyHostToDevice));
        CUCH(cudaMemcpy(v_device, v, (sizeof(double) * v_dim1 * v_dim2 * v_dim3), cudaMemcpyHostToDevice));
        });

        // Declare local variables
        double ax;
        double zero;
        double ay;
        double half;
        double az;

        // 3. Launch the CUDA Kernels
        measure_kernel_executions([&]() {
        zero = 0.0;
        half = 0.5;
        {
            // 3.1 Define execution configuration
        
            // Define the primary iteration space size for the kernel grid
            size_t total_elements = (((unz + 1) - 2 + 1) * ((uny + 1) - 2 + 1) * ((unx + 1) - 2 + 1));
        
            int threadsPerBlock = 256;
            int blocksPerGrid = (total_elements + threadsPerBlock - 1) / threadsPerBlock;
            
            int k_from = 2;
            int k_to = (unz + 1);
            
            int j_from = 2;
            int j_to = (uny + 1);
            
            int i_from = 2;
            int i_to = (unx + 1);
        
            // 4. Launch the CUDA kernel
            kernel_group_1_device<<<blocksPerGrid, threadsPerBlock>>>(
                unz,
                unx,
                zero,
                u2_device, u2_dim1, u2_dim2, u2_dim3,
                uny,
                k_from, k_to,
                j_from, j_to,
                i_from, i_to,
                total_elements
            );
        
            CUCH(cudaGetLastError());
        }
        ax = (0.25 / dxmin);
        ay = (0.25 / dymin);
        az = (0.25 / dzmin);
        {
            // 3.3 Define execution configuration
        
            // Define the primary iteration space size for the kernel grid
            size_t total_elements = (((unz + 1) - 2 + 1) * ((uny + 1) - 2 + 1) * ((unx + 1) - 2 + 1));
        
            int threadsPerBlock = 256;
            int blocksPerGrid = (total_elements + threadsPerBlock - 1) / threadsPerBlock;
            
            int k_from = 2;
            int k_to = (unz + 1);
            
            int j_from = 2;
            int j_to = (uny + 1);
            
            int i_from = 2;
            int i_to = (unx + 1);
        
            // 4. Launch the CUDA kernel
            kernel_group_3_device<<<blocksPerGrid, threadsPerBlock>>>(
                unz,
                az,
                w_device, w_dim1, w_dim2, w_dim3,
                ax,
                unx,
                v_device, v_dim1, v_dim2, v_dim3,
                u_device, u_dim1, u_dim2, u_dim3,
                u2_device, u2_dim1, u2_dim2, u2_dim3,
                ay,
                uny,
                k_from, k_to,
                j_from, j_to,
                i_from, i_to,
                total_elements
            );
        
            CUCH(cudaGetLastError());
        }
        ax = (0.5 / dxmin);
        ay = (0.125 / dymin);
        az = (0.125 / dzmin);
        {
            // 3.5 Define execution configuration
        
            // Define the primary iteration space size for the kernel grid
            size_t total_elements = (((unz + 1) - 2 + 1) * ((uny + 1) - 2 + 1) * ((unx + 1) - 2 + 1));
        
            int threadsPerBlock = 256;
            int blocksPerGrid = (total_elements + threadsPerBlock - 1) / threadsPerBlock;
            
            int k_from = 2;
            int k_to = (unz + 1);
            
            int j_from = 2;
            int j_to = (uny + 1);
            
            int i_from = 2;
            int i_to = (unx + 1);
        
            // 4. Launch the CUDA kernel
            kernel_group_5_device<<<blocksPerGrid, threadsPerBlock>>>(
                az,
                unz,
                ax,
                w_device, w_dim1, w_dim2, w_dim3,
                unx,
                v_device, v_dim1, v_dim2, v_dim3,
                u_device, u_dim1, u_dim2, u_dim3,
                ay,
                u2_device, u2_dim1, u2_dim2, u2_dim3,
                uny,
                k_from, k_to,
                j_from, j_to,
                i_from, i_to,
                total_elements
            );
        
            CUCH(cudaGetLastError());
        }
        {
            // 3.6 Define execution configuration
        
            // Define the primary iteration space size for the kernel grid
            size_t total_elements = (((unz + 1) - 2 + 1) * ((uny + 1) - 2 + 1) * ((unx + 1) - 2 + 1));
        
            int threadsPerBlock = 256;
            int blocksPerGrid = (total_elements + threadsPerBlock - 1) / threadsPerBlock;
            
            int k_from = 2;
            int k_to = (unz + 1);
            
            int j_from = 2;
            int j_to = (uny + 1);
            
            int i_from = 2;
            int i_to = (unx + 1);
        
            // 4. Launch the CUDA kernel
            kernel_group_6_device<<<blocksPerGrid, threadsPerBlock>>>(
                half,
                unz,
                unx,
                u2_device, u2_dim1, u2_dim2, u2_dim3,
                uny,
                k_from, k_to,
                j_from, j_to,
                i_from, i_to,
                total_elements
            );
        
            CUCH(cudaGetLastError());
        }
        });

        // Wait for GPU to finish
        CUCH(cudaDeviceSynchronize());

        size_t total_d2h_bytes = (sizeof(double) * u2_dim1 * u2_dim2 * u2_dim3);

        // 5. Copy results back from Device (GPU) to Host (CPU)
        measure_d2h(total_d2h_bytes, [&]() {
        CUCH(cudaMemcpy(u2, u2_device, (sizeof(double) * u2_dim1 * u2_dim2 * u2_dim3), cudaMemcpyDeviceToHost));
        });


        // 6. Free the GPU memory
        measure_free([&]() {
        CUCH(cudaFree(w_device));
        CUCH(cudaFree(u2_device));
        CUCH(cudaFree(u_device));
        CUCH(cudaFree(v_device));
        });
    }
}
}