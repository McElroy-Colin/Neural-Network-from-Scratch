// Header file for computation-related utility functions.

#ifndef COMPUTE_UTILS_H
#define COMPUTE_UTILS_H

/*
Return the maximum value in an array of unsigned integers. Assumes the array
has at least one element and that `size` accurately represents the length 
of `arr`.
*/
unsigned int arr_max(const unsigned int *arr, const unsigned int size);

/*
Multiply a flattened row-major matrix `A` by a vector `x` and store the result in `output`.
`A` must have dimensions `num_rows` X `num_cols` and the vectors `x` and `output` must be length `num_rows`.
Each thread computes a dot product on one row of the matrix with the vector. Assuming the vector is much larger than the number of threads, 
chunks of the vector are loaded into shared memory at a time and partial dot products are computed.

If there are significantly more threads than rows, this could be optimized such that multiple threads cooperate on the same row's partial dot.
*/
void gpu_matrix_multiply(const double *W, const unsigned int num_rows, const unsigned int num_cols, const double *x, double *output);

#endif