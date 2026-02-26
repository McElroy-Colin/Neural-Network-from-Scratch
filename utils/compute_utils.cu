// Source file for GPU-specific computation functions.

#include <stdarg.h> // needed?
#include <cuda_runtime.h>

#include "compute_utils.h"
#include "constants.h"


__device__ void gpu_matrix_multiply(
    const double *A, 
    const unsigned int num_rows, 
    const unsigned int num_cols, 
    const double *x, 
    double *output) {
    
    // Designate a tile of shared memory for the vector.
    __shared__ double tile[TILE_SIZE];
    
    unsigned int row = blockIdx.x*blockDim.x + threadIdx.x;
    double sum = 0.0;

    // Loop over tiles of the input vector
    for (unsigned int t = 0; t < (num_cols + TILE_SIZE - 1)/TILE_SIZE; ++t) {
        // Load a tile of `x` into shared memory
        unsigned int col = t*TILE_SIZE + threadIdx.y;
        if ((col < num_cols) && (threadIdx.x == 0)) {
            tile[threadIdx.y] = x[col];
        }

        __syncthreads(); // Ensure tile is loaded

        if (row < num_rows) {
            for (unsigned int i = 0; i < TILE_SIZE; ++i) {
                unsigned int c = t*TILE_SIZE + i;
                if (c < num_cols) {
                    sum += A[row*num_cols + c]*tile[i];
                }
            }
        }

        __syncthreads(); // Ensure all threads are done using shared memory
    }

    // Write result to global memory
    if (row < num_rows) {
        output[row] = sum;
    }
}