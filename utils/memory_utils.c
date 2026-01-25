// Source file for memory-related utility functions.

#include <stdlib.h>
#include <stdarg.h>

#include "memory_utils.h"


void free_2d_darr(double **arr, const unsigned int num_allocated) {
    for (int i = 0; i < num_allocated; i++) {
        free(arr[i]);
    }
    free(arr);
}
void free_2d_carr(char **arr, const unsigned int num_allocated) {
    for (int i = 0; i < num_allocated; i++) {
        free(arr[i]);
    }
    free(arr);
}

void free_ptrs(void *p1, ...) {
    va_list ptrs;
    void *curr_ptr = p1;

    va_start(ptrs, curr_ptr);
    while (curr_ptr != NULL) {
        free(curr_ptr);
        curr_ptr = va_arg(ptrs, void*);
    }

    va_end(ptrs);
}