// Header file for computation-related utility functions.

#ifndef COMPUTE_UTILS_H
#define COMPUTE_UTILS_H

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
Compute an array of layer offsets representing the number of weights up to a given layer in the neural network.
    e.g. if `layer_offsets[2]` was `256`, then `weights[256]` would be the first weight of layer 1.

Parameters:
    `num_layers`: The number of layers in the neural network excluding the input (feature) layer. (input)
    `num_features`: Length of the network's feature layer. (input)
    `layer_sizes`: Array of integers representing the size of each layer, excluding the input (feature) layer. (input)
    `layer_offsets`: Allocated buffer to be filled with the layer offset values. (output)

    Returns -1 if there are not at least two layers, otherwise 0.
*/
int compute_layer_offsets(const unsigned int num_layers, 
    const unsigned int num_features, 
    const unsigned int *layer_sizes, 
    unsigned int *layer_offsets
);

#endif