
#include <stdarg.h>
#include <cuda_runtime.h>

#include "memory_utils.h"



void cudafree_2d_darr(double **arr, const unsigned int num_allocated) {
    for (int i = 0; i < num_allocated; i++) {
        cudaFree(arr[i]);
    }
    cudaFree(arr);
}

void cudafree_ptrs(void *p1, ...) {
    va_list ptrs;
    void *curr_ptr = p1;

    va_start(ptrs, curr_ptr);
    while (curr_ptr != NULL) {
        cudaFree(curr_ptr);
        curr_ptr = va_arg(ptrs, void*);
    }

    va_end(ptrs);
}