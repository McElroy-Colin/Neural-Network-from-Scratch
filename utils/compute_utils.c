// Source file for computation-related utility functions.

#include <stdio.h>
#include <stdlib.h>

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

unsigned int* compute_weight_offsets(const unsigned int num_layers, 
    const unsigned int num_features, 
    const unsigned int *layers
) {
    if (num_layers < 2) {
        fprintf(stderr, "from compute_weight_offsets(), less than 2 layers\n");
        return NULL;
    }

    unsigned int *weight_offsets = malloc(num_layers*sizeof(unsigned int));
    if (!weight_offsets) {
        fprintf(stderr, "from compute_weight_offsets(), memory allocation error\n");
        return NULL;
    }

    weight_offsets[0] = 0;
    weight_offsets[1] = num_features*layers[0];
    for (int i = 2; i < num_layers; i++) {
        weight_offsets[i] = weight_offsets[i - 1] + layers[i - 1]*layers[i];
    }

    return weight_offsets;
}

unsigned int* compute_bias_offsets(const unsigned int num_layers, 
    const unsigned int *layers
) {
    if (num_layers < 1) {
        fprintf(stderr, "from compute_bias_offsets(), 0 layers\n");
        return NULL;
    }

    unsigned int *bias_offsets = malloc(num_layers*sizeof(unsigned int));
    if (!bias_offsets) {
        fprintf(stderr, "from compute_bias_offsets(), memory allocation error\n");
        return NULL;
    }

    bias_offsets[0] = 0;
    for (int i = 1; i < num_layers; i++) {
        bias_offsets[i] = layers[i - 1] + bias_offsets[i - 1];
    }

    return bias_offsets;
}
