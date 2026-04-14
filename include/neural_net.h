// Header file for the primary neural network functions.

#ifndef NEURAL_NET_H
#define NEURAL_NET_H

#include "activation_functions.h"


/*
Perform a serial feed-forward dense neural network computation on the host given relevant parameters.
This function assumes that all vector/matrix dimensionality is correct. 
    e.g. weight matrices should have the correct dimensionality for their previous and current layer sizes.
Note, the output values will be in `buffer1` after the function call.

Parameters:
    `features`: input feature vector (input)
    `num_features`: length of `features` (input)
    `layers`: array of layer sizes where the size of the array is the number of layers excluding the input vector (input)
                e.g. If `layers[2]` is 5, then the THIRD layer after input has FIVE neurons.
    `num_layers`: number of layers in the network excluding an input vector; also the length of `layers` (input)
    `layer_offsets`: array containing the offset value for each layer of the network.
        e.g. if `layer_offsets[2]` was `256`, then `weights[256]` would be the first weight of layer 1.
    `activation_fns`: An array of function enum values spercifying the activation for its respective layer,
                    so `activation_fns` is also length `num_layers`. (input)
                        e.g. If `activation_fns[3]` references a sigmoid function, then layer FOUR will use sigmoid on its neurons.
    `weights`: An array of flattened weight matrices where each matrix applies to its respective layer. So, the first `num_features*layers[0]` values
               in weights correspond to the matrix connecting the feature vector to the first hidden layer of the neural network. (input)
    `biases`: A flattened array of bias vectors where each vector applies to its respective layer, i.e. the first `layers[0]` of `biases` corresponds
              to the biases for the neuron values in the first layer. (input)
    `buffer1/2`: Buffers large enough to hold any given layer of the network including a feature vector. (input/output)

Returns -1 on error, otherwise 0.
*/
int feed_forward_hst(const double *features,
    const unsigned int num_features,  
    const unsigned int *layers, 
    const unsigned int num_layers,
    const unsigned int *layer_offsets,
    const ActivationFunc *activation_fns,
    const double *weights, 
    const double *biases,
    double *buffer1, double *buffer2
);



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