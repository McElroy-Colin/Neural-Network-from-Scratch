// Source file for memory-related utility functions.

#include <stdlib.h>

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