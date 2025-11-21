// Source file for computation-related utility functions.

#include "compute_utils.h"


unsigned int arr_max(const unsigned int *arr, const unsigned int size) {
    double curr_max = arr[0];

    for (int i = 1; i < size; i++) {
        if (arr[i] > curr_max) {
            curr_max = arr[i];
        }
    }

    return curr_max;
}
