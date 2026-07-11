// Header file for computation-related utility functions.

#ifndef COMPUTE_UTILS_H
#define COMPUTE_UTILS_H

// Used in both C and CUDA versions, so define them as C++ code.
#ifdef __cplusplus
extern "C"
#endif
/*
Return the maximum value in an array of unsigned integers. Assumes the array
has at least one element and that `size` accurately represents the length 
of `arr`.
*/
unsigned int arr_max(const unsigned int *arr, const unsigned int size);

/*
Compute an array of weight offsets representing the number of weights up to a given layer in the neural network.
    e.g. if `weight_offsets[2]` was `256`, then `weights[256]` would be the first weight of layer 1.
The returned ponted must be freed if there is no error.

Parameters:
    `num_layers`: The number of layers in the neural network excluding the input (feature) layer. (input)
    `num_features`: Length of the network's feature layer. (input)
    `layers`: Array of integers representing the size of each layer, excluding the input (feature) layer. (input)

    Returns a pointer to the newly allocated offset array, or `NULL` on an error.
*/
unsigned int* compute_weight_offsets(const unsigned int num_layers, 
    const unsigned int num_features, 
    const unsigned int *layers
);

/*
Compute an array of bias offsets representing the number of biases up to a given layer in the neural network.
    e.g. if `bias_offsets[2]` was `32`, then `biases[32]` would be the first bias of layer 1.
The returned ponted must be freed if there is no error.

Parameters:
    `num_layers`: The number of layers in the neural network excluding the input (feature) layer. (input)
    `layers`: Array of integers representing the size of each layer, excluding the input (feature) layer. (input)

    Returns a pointer to the newly allocated offset array, or `NULL` on an error.
*/
unsigned int* compute_bias_offsets(const unsigned int num_layers, const unsigned int *layers);

#endif