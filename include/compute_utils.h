// Header file for computation-related utility functions.

#ifndef COMPUTE_UTILS_H
#define COMPUTE_UTILS_H

/*
Return the maximum value in an array of unsigned integers. Assumes the array
has at least one element and that `size` accurately represents the length 
of `arr`.
*/
unsigned int arr_max(const unsigned int *arr, const unsigned int size);

#ifdef __CUDACC__
/*
Matrix vector multiply-add operation on the device.
Multiply a flattened row-major matrix `A` by a vector `x` and add a vector `y` to the result. Store the final result in `output`.
`A` must have dimensions `num_rows` X `num_cols` and the vectors `x` and `output` must be length `num_rows`.
Each thread computes a dot product on one row of the matrix with the vector. Assuming the vector is much larger than the number of threads, 
chunks of the vector are loaded into shared memory at a time and partial dot products are computed.\

Grid distribution should use 3D cooperative loading with the `TILE_SIZE` constant in `constants.h`.
`TILE_SIZE` determines the amount of the vector `x` that is loaded into the device's shared memory at once.
To call the kernel with the value `TILE_SIZE` and a matrix `A` that has dimension `n x m`, do
    ```
    dim3 num_blocks((n + TILE_SIZE - 1) / TILE_SIZE, 1); // ceiling division
    dim3 threads_per_block(TILE_SIZE, TILE_SIZE);
    dvc_matrixvector_mad<<<num_blocks, threads_per_block>>>(A, n, m, x, y, output);
    ```
Note, the `TILE_SIZE` threads in the y-dimension of each block serve to simultaneously load the vector `x` into shared memory.

If there are significantly more threads than rows, this could be optimized such that multiple threads cooperate on the same row's partial dot.
*/
__global__ void dvc_matrixvector_mad(const double *W, const unsigned int num_rows, const unsigned int num_cols, const double *x, const double *y, double *output);

#endif

#endif