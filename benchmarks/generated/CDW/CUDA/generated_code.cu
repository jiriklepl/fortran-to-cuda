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
    int wnz,
    int wnx,
    double zero,
    double* __restrict__ w2, size_t w2_dim1, size_t w2_dim2, size_t w2_dim3,
    int wny,
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
    w2[F_IDX(i, j, k, w2_dim1, w2_dim2, w2_dim3)] = zero;
    
}

__global__ 
void kernel_group_3_device(
    int wnz,
    double az,
    double* __restrict__ w, size_t w_dim1, size_t w_dim2, size_t w_dim3,
    double ax,
    int wnx,
    double* __restrict__ v, size_t v_dim1, size_t v_dim2, size_t v_dim3,
    double* __restrict__ u, size_t u_dim1, size_t u_dim2, size_t u_dim3,
    double* __restrict__ w2, size_t w2_dim1, size_t w2_dim2, size_t w2_dim3,
    double ay,
    int wny,
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
    w2[F_IDX(i, j, k, w2_dim1, w2_dim2, w2_dim3)] = (-(((((((az * ((w[F_IDX(i, j, (k + 1), w_dim1, w_dim2, w_dim3)] + w[F_IDX(i, j, k, w_dim1, w_dim2, w_dim3)]))) * ((w[F_IDX(i, j, (k + 1), w_dim1, w_dim2, w_dim3)] + w[F_IDX(i, j, k, w_dim1, w_dim2, w_dim3)]))) - ((az * ((w[F_IDX(i, j, k, w_dim1, w_dim2, w_dim3)] + w[F_IDX(i, j, (k - 1), w_dim1, w_dim2, w_dim3)]))) * ((w[F_IDX(i, j, k, w_dim1, w_dim2, w_dim3)] + w[F_IDX(i, j, (k - 1), w_dim1, w_dim2, w_dim3)]))))) + ((((ay * ((w[F_IDX(i, (j + 1), k, w_dim1, w_dim2, w_dim3)] + w[F_IDX(i, j, k, w_dim1, w_dim2, w_dim3)]))) * ((v[F_IDX(i, j, (k + 1), v_dim1, v_dim2, v_dim3)] + v[F_IDX(i, j, k, v_dim1, v_dim2, v_dim3)]))) - ((ay * ((w[F_IDX(i, j, k, w_dim1, w_dim2, w_dim3)] + w[F_IDX(i, (j - 1), k, w_dim1, w_dim2, w_dim3)]))) * ((v[F_IDX(i, (j - 1), k, v_dim1, v_dim2, v_dim3)] + v[F_IDX(i, (j - 1), (k + 1), v_dim1, v_dim2, v_dim3)])))))) + ((((ax * ((w[F_IDX((i + 1), j, k, w_dim1, w_dim2, w_dim3)] + w[F_IDX(i, j, k, w_dim1, w_dim2, w_dim3)]))) * ((u[F_IDX(i, j, (k + 1), u_dim1, u_dim2, u_dim3)] + u[F_IDX(i, j, k, u_dim1, u_dim2, u_dim3)]))) - ((ax * ((w[F_IDX(i, j, k, w_dim1, w_dim2, w_dim3)] + w[F_IDX((i - 1), j, k, w_dim1, w_dim2, w_dim3)]))) * ((u[F_IDX((i - 1), j, (k + 1), u_dim1, u_dim2, u_dim3)] + u[F_IDX((i - 1), j, k, u_dim1, u_dim2, u_dim3)]))))))));
    
}

__global__ 
void kernel_group_5_device(
    double az,
    int wnz,
    double ax,
    double* __restrict__ w, size_t w_dim1, size_t w_dim2, size_t w_dim3,
    int wnx,
    double* __restrict__ v, size_t v_dim1, size_t v_dim2, size_t v_dim3,
    double* __restrict__ u, size_t u_dim1, size_t u_dim2, size_t u_dim3,
    double ay,
    double* __restrict__ w2, size_t w2_dim1, size_t w2_dim2, size_t w2_dim3,
    int wny,
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
    double uadv;
    double vadv;

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
    uadv = ((((u[F_IDX(i, j, k, u_dim1, u_dim2, u_dim3)] + u[F_IDX(i, j, (k + 1), u_dim1, u_dim2, u_dim3)]) + u[F_IDX((i - 1), j, k, u_dim1, u_dim2, u_dim3)]) + u[F_IDX((i - 1), j, (k + 1), u_dim1, u_dim2, u_dim3)]));
    vadv = ((((v[F_IDX(i, j, k, v_dim1, v_dim2, v_dim3)] + v[F_IDX(i, j, (k + 1), v_dim1, v_dim2, v_dim3)]) + v[F_IDX(i, (j - 1), k, v_dim1, v_dim2, v_dim3)]) + v[F_IDX(i, (j - 1), (k + 1), v_dim1, v_dim2, v_dim3)]));
    w2[F_IDX(i, j, k, w2_dim1, w2_dim2, w2_dim3)] = (w2[F_IDX(i, j, k, w2_dim1, w2_dim2, w2_dim3)] - (((((ax * ((w[F_IDX((i + 1), j, k, w_dim1, w_dim2, w_dim3)] - w[F_IDX((i - 1), j, k, w_dim1, w_dim2, w_dim3)]))) * uadv) + ((ay * ((w[F_IDX(i, (j + 1), k, w_dim1, w_dim2, w_dim3)] - w[F_IDX(i, (j - 1), k, w_dim1, w_dim2, w_dim3)]))) * vadv)) + ((az * ((w[F_IDX(i, j, (k + 1), w_dim1, w_dim2, w_dim3)] - w[F_IDX(i, j, (k - 1), w_dim1, w_dim2, w_dim3)]))) * w[F_IDX(i, j, k, w_dim1, w_dim2, w_dim3)]))));
    
}

__global__ 
void kernel_group_6_device(
    double half,
    int wnz,
    int wnx,
    double* __restrict__ w2, size_t w2_dim1, size_t w2_dim2, size_t w2_dim3,
    int wny,
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
    w2[F_IDX(i, j, k, w2_dim1, w2_dim2, w2_dim3)] = (w2[F_IDX(i, j, k, w2_dim1, w2_dim2, w2_dim3)] * half);
    
}


// The wrapper function called by Fortran
extern "C" {
    void cpp_start_hot() {
        reset_timing_vectors();
    }

    void cpp_finish_hot() {
        print_timing_summary();
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
        #ifdef USE_PINNED_MEMORY

        if (pinned_ptrs.find(w) == pinned_ptrs.end()) {
            CUCH(cudaHostRegister(w, (sizeof(double) * w_dim1 * w_dim2 * w_dim3), cudaHostRegisterPortable));
            pinned_ptrs.insert(w);
        }
        if (pinned_ptrs.find(v) == pinned_ptrs.end()) {
            CUCH(cudaHostRegister(v, (sizeof(double) * v_dim1 * v_dim2 * v_dim3), cudaHostRegisterPortable));
            pinned_ptrs.insert(v);
        }
        if (pinned_ptrs.find(w2) == pinned_ptrs.end()) {
            CUCH(cudaHostRegister(w2, (sizeof(double) * w2_dim1 * w2_dim2 * w2_dim3), cudaHostRegisterPortable));
            pinned_ptrs.insert(w2);
        }
        if (pinned_ptrs.find(u) == pinned_ptrs.end()) {
            CUCH(cudaHostRegister(u, (sizeof(double) * u_dim1 * u_dim2 * u_dim3), cudaHostRegisterPortable));
            pinned_ptrs.insert(u);
        }

        #endif // USE_PINNED_MEMORY

        // 1. Allocate memory on the GPU (Device)
        double* w_device;
        double* v_device;
        double* w2_device;
        double* u_device;

        measure_alloc([&]() {
        CUCH(cudaMalloc(&w_device, (sizeof(double) * w_dim1 * w_dim2 * w_dim3)));
        CUCH(cudaMalloc(&v_device, (sizeof(double) * v_dim1 * v_dim2 * v_dim3)));
        CUCH(cudaMalloc(&w2_device, (sizeof(double) * w2_dim1 * w2_dim2 * w2_dim3)));
        CUCH(cudaMalloc(&u_device, (sizeof(double) * u_dim1 * u_dim2 * u_dim3)));
        });

        size_t total_h2d_bytes = (sizeof(double) * w_dim1 * w_dim2 * w_dim3) + (sizeof(double) * v_dim1 * v_dim2 * v_dim3) + (sizeof(double) * w2_dim1 * w2_dim2 * w2_dim3) + (sizeof(double) * u_dim1 * u_dim2 * u_dim3);

        // 2. Copy inputs from Host (CPU) to Device (GPU)
        measure_h2d(total_h2d_bytes, [&]() {
        CUCH(cudaMemcpy(w_device, w, (sizeof(double) * w_dim1 * w_dim2 * w_dim3), cudaMemcpyHostToDevice));
        CUCH(cudaMemcpy(v_device, v, (sizeof(double) * v_dim1 * v_dim2 * v_dim3), cudaMemcpyHostToDevice));
        CUCH(cudaMemcpy(u_device, u, (sizeof(double) * u_dim1 * u_dim2 * u_dim3), cudaMemcpyHostToDevice));
        });

        // Declare local variables
        double az;
        double ay;
        double ax;
        double zero;
        double half;

        // 3. Launch the CUDA Kernels
        measure_kernel_executions([&]() {
        zero = 0.0;
        half = 0.5;
        {
            // 3.1 Define execution configuration
        
            // Define the primary iteration space size for the kernel grid
            size_t total_elements = (((wnz + 1) - 2 + 1) * ((wny + 1) - 2 + 1) * ((wnx + 1) - 2 + 1));
        
            int threadsPerBlock = 256;
            int blocksPerGrid = (total_elements + threadsPerBlock - 1) / threadsPerBlock;
            
            int k_from = 2;
            int k_to = (wnz + 1);
            
            int j_from = 2;
            int j_to = (wny + 1);
            
            int i_from = 2;
            int i_to = (wnx + 1);
        
            // 4. Launch the CUDA kernel
            kernel_group_1_device<<<blocksPerGrid, threadsPerBlock>>>(
                wnz,
                wnx,
                zero,
                w2_device, w2_dim1, w2_dim2, w2_dim3,
                wny,
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
            size_t total_elements = (((wnz + 1) - 2 + 1) * ((wny + 1) - 2 + 1) * ((wnx + 1) - 2 + 1));
        
            int threadsPerBlock = 256;
            int blocksPerGrid = (total_elements + threadsPerBlock - 1) / threadsPerBlock;
            
            int k_from = 2;
            int k_to = (wnz + 1);
            
            int j_from = 2;
            int j_to = (wny + 1);
            
            int i_from = 2;
            int i_to = (wnx + 1);
        
            // 4. Launch the CUDA kernel
            kernel_group_3_device<<<blocksPerGrid, threadsPerBlock>>>(
                wnz,
                az,
                w_device, w_dim1, w_dim2, w_dim3,
                ax,
                wnx,
                v_device, v_dim1, v_dim2, v_dim3,
                u_device, u_dim1, u_dim2, u_dim3,
                w2_device, w2_dim1, w2_dim2, w2_dim3,
                ay,
                wny,
                k_from, k_to,
                j_from, j_to,
                i_from, i_to,
                total_elements
            );
        
            CUCH(cudaGetLastError());
        }
        ax = (0.125 / dxmin);
        ay = (0.125 / dymin);
        az = (0.5 / dzmin);
        {
            // 3.5 Define execution configuration
        
            // Define the primary iteration space size for the kernel grid
            size_t total_elements = (((wnz + 1) - 2 + 1) * ((wny + 1) - 2 + 1) * ((wnx + 1) - 2 + 1));
        
            int threadsPerBlock = 256;
            int blocksPerGrid = (total_elements + threadsPerBlock - 1) / threadsPerBlock;
            
            int k_from = 2;
            int k_to = (wnz + 1);
            
            int j_from = 2;
            int j_to = (wny + 1);
            
            int i_from = 2;
            int i_to = (wnx + 1);
        
            // 4. Launch the CUDA kernel
            kernel_group_5_device<<<blocksPerGrid, threadsPerBlock>>>(
                az,
                wnz,
                ax,
                w_device, w_dim1, w_dim2, w_dim3,
                wnx,
                v_device, v_dim1, v_dim2, v_dim3,
                u_device, u_dim1, u_dim2, u_dim3,
                ay,
                w2_device, w2_dim1, w2_dim2, w2_dim3,
                wny,
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
            size_t total_elements = (((wnz + 1) - 2 + 1) * ((wny + 1) - 2 + 1) * ((wnx + 1) - 2 + 1));
        
            int threadsPerBlock = 256;
            int blocksPerGrid = (total_elements + threadsPerBlock - 1) / threadsPerBlock;
            
            int k_from = 2;
            int k_to = (wnz + 1);
            
            int j_from = 2;
            int j_to = (wny + 1);
            
            int i_from = 2;
            int i_to = (wnx + 1);
        
            // 4. Launch the CUDA kernel
            kernel_group_6_device<<<blocksPerGrid, threadsPerBlock>>>(
                half,
                wnz,
                wnx,
                w2_device, w2_dim1, w2_dim2, w2_dim3,
                wny,
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

        size_t total_d2h_bytes = (sizeof(double) * w2_dim1 * w2_dim2 * w2_dim3);

        // 5. Copy results back from Device (GPU) to Host (CPU)
        measure_d2h(total_d2h_bytes, [&]() {
        CUCH(cudaMemcpy(w2, w2_device, (sizeof(double) * w2_dim1 * w2_dim2 * w2_dim3), cudaMemcpyDeviceToHost));
        });


        // 6. Free the GPU memory
        measure_free([&]() {
        CUCH(cudaFree(w_device));
        CUCH(cudaFree(v_device));
        CUCH(cudaFree(w2_device));
        CUCH(cudaFree(u_device));
        });
    }
}
}