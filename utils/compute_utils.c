// Source file for computation-related utility functions.

#include <stdio.h>

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

int compute_layer_offsets(const unsigned int num_layers, 
    const unsigned int num_features, 
    const unsigned int *layer_sizes, 
    unsigned int *layer_offsets
) {
    if (num_layers < 2) {
        fprintf(stderr, "from compute_layer_offsets(), less than 2 layers\n");
        return -1;
    }

    layer_offsets[0] = 0;
    layer_offsets[1] = num_features*layer_sizes[0];
    for (int i = 2; i < num_layers; i++) {
        layer_offsets[i] = layer_offsets[i - 1] + layer_sizes[i - 1]*layer_sizes[i];
    }

    return 0;
}
