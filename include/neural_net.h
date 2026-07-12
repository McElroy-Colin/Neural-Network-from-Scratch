// Header file for the primary neural network functions.

#ifndef NEURAL_NET_H
#define NEURAL_NET_H

#include "activation_functions.h"


// Neural Network structure containing all relevent network topology.
typedef struct {
    // Array of layer sizes where the size of the array is the number of layers, excluding the input vector, length 'num_layers'.
    unsigned int *layers;

    // An array of flattened row-major weight matrices where each matrix applies to its respective layer.
    // So, the first `num_features*layers[0]` values in `weights` correspond to the matrix connecting the feature 
    // vector to the first hidden layer of the neural network.
    double *weights;

    // A flattened array of bias vectors where each vector applies to its respective layer. 
    //     i.e. the first `layers[0]` values in `biases` corresponds to the biases for the first hidden layer layer.
    double *biases;

    // An array containing the weight matrix offset value for each layer of the network.
    //     e.g. if `weight_offsets[2]` was `256`, then `weights[256]` would be the first weight of the second hidden layer.
    unsigned int *weight_offsets; // TODO: name change everywhere

    // An array containing the bias vector offset value for each layer of the network.
    //     e.g. if `bias_offsets[2]` was `96`, then `biases[96]` would be the first bias of the second hidden layer.
    unsigned int *bias_offsets;

    // An array of function enum values specifying the activation for its respective layer, length 'num_layers'
    //     e.g. If `activation_fns[3]` references a sigmoid function, then layer FOUR will use sigmoid.
    ActivationFunc *activation_fns;

    // Length of the input feature vector. This is NOT included in `layers`.
    unsigned int num_features;
    
    // Number of layers in the network excluding the input vector but including the output vector.
    unsigned int num_layers;

    // Size of the largest layer in the network including the input and output vectors.
    //     i.e. `max_layer_size = max(max(layers), num_features)`
    unsigned int max_layer_size;

    // Total number of weight values across all layers in the network.
    unsigned int num_weights;

    // Total number of bias values across all layers in the network. Also the total number of non-input neurons.
    //     i.e. `num_biases = sum(layers)`
    unsigned int num_biases;
} NeuralNetwork;

// Used in both C and CUDA versions, so define it as C++ code.
#ifdef __cplusplus
extern "C" {
#endif

// Initialize all variables of a neural network object given the necessary parameters.
// Returns the created neural network object, or `NULL` on error.
// Note, all arrays in the neural network object must be freed on a successful call.
int init_neural_net(
    unsigned int num_features,
    unsigned int *layers,
    unsigned int num_layers,
    double *weights,
    unsigned int num_weights,
    double *biases,
    unsigned int num_biases,
    ActivationFunc *activation_fns,
    NeuralNetwork *output_nn
);

// TODO: make a destructor once main has heap allocated arrays for its neural net.
void free_neural_net(NeuralNetwork *neural_net);

#ifdef __cplusplus
}
#endif

/*
Perform a serial feed-forward dense neural network computation on the host given a neural network.
This function assumes that all vector/matrix dimensionality is correct in the neural network object. 
    e.g. weight matrices should have the correct dimensionality for their previous and current layer sizes.
Assume that `buffer1` holds the initial feature vector before the function call.
Similarly, `buffer1` will hold the resulting output vector values when after the function call.

Parameters:
    `neural_net`: Neural network object containing its topology. (input)
    `buffer1/2`: Buffers large enough to hold any given layer of the network including a feature vector. (input/output)
                 `buffer1` also holds the initial feature vector before the function call as well as the output vector after the function call.
*/
void feed_forward_hst(
    const NeuralNetwork *neural_net,
    double *buffer1, double *buffer2
);

// Compute and return the Mean-Squared Error (MSE) between two vectors `y` and `y_hat`.
// This function assumes that both vectors are length `count`.
double mean_squared_error(
    const double *y, 
    const double *y_hat, 
    const unsigned int count
);

/*
Compute the MSE for a batch of train-test vector pairs. Store the results in the `errs` matrix.
This function assumes that each vector in the `ys` and `y_hats` matrices is length `count_per_vec`, and 
that there are `count_per_vec*num_vecs` total elements in each.

Parameters:
    `ys`: Row-major matrix of true label vectors, where each `count_per_vec` elements is a vector with
          a total of `num_vecs` vectors. (input)
    `y_hats`: Same as `ys`, but for observed vectors. (input)
    `count_per_vec`: Length of each vector in the above matrices. (input)
    `num_vecs`: Number of vectors in the above matrices. (input)
    `errs`: Allocated vector of length `num_vecs` to hold each MSE value. (output)

*/
void batch_mse(
    const double *ys, 
    const double *y_hats, 
    const unsigned int count_per_vec,
    const unsigned int num_vecs,
    double *errs
);

#endif