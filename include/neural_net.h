// Header file for the primary neural network functions.

#ifndef NEURAL_NET_H
#define NEURAL_NET_H

#include "activation_functions.h"


/*
Perform a serial feed-forward dense neural network computation on the host given relevant parameters.
This function assumes that all vector/matrix dimensionality is correct. 
    e.g. weight matrices should have the correct dimensionality for their previous and current layer sizes.

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
    `buffer1/2`: Buffers large enough to hold any given layer of the network including a feature vector. (input)
    `output`: Pointer to an array to hold the final output of the neural network. `*output` should be length `layers[num_layers - 1]`. (output) 

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
    double *buffer1, double *buffer2,
    double **output
);


#ifdef __CUDACC__
/*
Perform a parallel feed-forward dense neural network computation given on the device given relevant parameters.
This function assumes that all vector/matrix dimensionality is correct. 
    e.g. weight matrices should have the correct dimensionality for their previous and current layer sizes.

Parameters:
    `features`: input feature vector (input)
    `num_features`: length of `features` (input)
    `layers`: array of layer sizes where the size of the array is the number of layers excluding the input vector (input)
                e.g. If `layers[2] = 5`, then the THIRD layer after input has FIVE neurons.
    `num_layers`: number of layers in the network excluding an input vector; also the length of `layers` (input)
    `layer_offsets`: array containing the offset value for each layer of the network. (input)
                        e.g. If `layer_offsets[3] = 256`, then `weights[256]` would be the first weight of layer 2.
    `activation_fns`: An array of function enum values spercifying the activation for its respective layer,
                      so `activation_fns` is also length `num_layers`. (input)
                        e.g. If `activation_fns[3]` references a sigmoid function, then layer FOUR will use sigmoid on its neurons.
    `weights`: An array of flattened weight matrices in device memory where each matrix applies to its respective layer. 
               So, the first `num_features*layers[0]` values in weights correspond to the matrix connecting the 
               feature vector to the first hidden layer of the neural network. (input)
    `biases`: A flattened array of bias vectors in device memory where each vector applies to its respective layer, 
                  i.e. the first `layers[0]` of `biases` corresponds to the biases for the neuron values in the first layer. (input)
    `dvc_buffer1/2`: Buffers on the device large enough to hold any given layer of the network including a feature vector. (input)
    `output`: Pointer to an array to hold the final output of the neural network. `*output` should be length `layers[num_layers - 1]`. (output) 

Returns -1 on error, otherwise 0.
*/
__global__ void feed_forward_dvc(const unsigned int num_features,  
    const unsigned int *layers, 
    const unsigned int num_layers,
    const unsigned int *layer_offsets,
    const unsigned int max_layer_size,
    const ActivationFunc *activation_fns,
    const double *weights, 
    const double *biases,
    double *buffer1, double *buffer2,
    double **output
);

#endif

/*
Copy the given neural network to the device and assign output pointers to the device memory locations.

Parameters:
    `max_layer`: Size of the largest layer in the network, including the input (feature) layer. (input)
    `num_layers`: number of layers in the network excluding an input vector; also the length of `layers` (input)
    `layer_offsets`: array containing the offset value for each layer of the network.
        e.g. if `layer_offsets[2]` was `256`, then `weights[256]` would be the first weight of layer 1.
    `activation_fns`: An array of function enum values spercifying the activation for its respective layer,
                    so `activation_fns` is also length `num_layers`. (input)
                        e.g. If `activation_fns[3]` references a sigmoid function, then layer FOUR will use sigmoid on its neurons.
    `weights`: An array of flattened weight matrices where each matrix applies to its respective layer. So, the first `num_features*layers[0]` values
               in `weights` correspond to the matrix connecting the feature vector to the first hidden layer of the neural network. (input)
    `num_weights`: Total number of weights in the network. (input)
    `biases`: An array of flattened bias vectors where each vector applies to its respective layer. (input)
                e.g. The first `layers[0]` values of `biases` correspond to the biases for the first layer of the network. (input)
    `num_biases`: Total number of biases in the network. (input)
    `dvc_weights`: Unallocated pointer that will point to the first weight stored in the device. (output)
    `dvc_biases`: Unallocated pointer that will point to the first bias stored in the device. (output)
    `dvc_layer_offsets`: Unallocated pointer that will point to the first layer offset value stored in the device. (output)
    `dvc_buffer1/2`: Buffers on the device large enough to hold any given layer of the network including a feature vector. (output)

Returns -1 on error, otherwise 0.
*/
int nn_load_dvc(const unsigned int max_layer,
    const unsigned int num_layers,
    const unsigned int *layer_offsets,
    const ActivationFunc *activation_fns,
    const double *weights,
    const unsigned int num_weights,
    const double *biases,
    const unsigned int num_biases,
    double **dvc_weights,
    double **dvc_layer_offsets,
    double **dvc_biases,
    ActivationFunc **dvc_activation_fns,
    double **dvc_buffer1, double **dvc_buffer2
);

#endif