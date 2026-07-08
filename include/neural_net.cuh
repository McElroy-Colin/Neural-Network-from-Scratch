#ifndef NEURAL_NET_CUH
#define NEURAL_NET_CUH



#include "activation_functions.cuh"
#include "neural_net.h"

/*
Perform a parallel feed-forward dense neural network computation given on the device given relevant parameters.
This function assumes that all vector/matrix dimensionality is correct. 
    e.g. weight matrices should have the correct dimensionality for their previous and current layer sizes.
Assume that `buffer1` holds the initial feature vector on the device when the kernel launches.
Similarly, `buffer1` will hold the resulting output vector values when after the function call.

Grid distribution uses 3D cooperative loading with the `TILE_SIZE` constant in `constants.h`.
`TILE_SIZE` determines the amount of any given vector that is loaded into an SM's shared memory at once.
To call the kernel with the value `TILE_SIZE` and a maximum layer size of `n`, do
    ```
    dim3 num_blocks((n + TILE_SIZE - 1) / TILE_SIZE, 1); // ceiling division
    dim3 threads_per_block(TILE_SIZE, TILE_SIZE);
    feed_forward_dvc<<<num_blocks, threads_per_block>>>( ... );
    ```
Note, the `TILE_SIZE` threads in the y-dimension of each block are for cooperative loadng of any given layer into shared memory.

Parameters:
    `num_features`: length of `features` (input)
    `layers`: array of layer sizes in device memory where the size of the array is the number of layers excluding the input vector (input)
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
    `buffer1/2`: Buffers on the device large enough to hold any given layer of the network including a feature vector. (input/output)
                     `buffer1` also holds the initial feature vector before the function call as well as the output vector after the function call,
                     both on device memory. 
*/
__global__ void feed_forward_dvc(const unsigned int num_features,  
    const unsigned int *layers, 
    const unsigned int num_layers,
    const unsigned int *layer_offsets,
    const ActivationFunc *activation_fns,
    const double *weights, 
    const double *biases,
    double *buffer1, double *buffer2
);

/*
Copy the given neural network to the device and assign output pointers to the device memory locations.

Parameters:
    `hst_neural_net`: Fully initialized neural network object in host memory. All values should be initialized and accurate. (input)
    `shell_neural_net`: Uninitialized nerual network object in host memory. All internal pointers will point to device memory after the call. (input/output)
    `dvc_buffer1/2`: Buffers on the device large enough to hold any given layer of the network including a feature vector. (output)

Returns -1 on error, otherwise 0.
*/
int nn_load_dvc(
    NeuralNetwork *hst_neural_net,
    NeuralNetwork *shell_neural_net,
    double **dvc_buffer1, double **dvc_buffer2
);

#endif