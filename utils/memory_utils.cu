
#include <cuda_runtime.h>

#include "memory_utils.h"



void cudafree_2d_darr(double **arr, const unsigned int num_allocated) {
    for (int i = 0; i < num_allocated; i++) {
        cudaFree(arr[i]);
    }
    cudaFree(arr);
}