// Header file for the primary neural network functions.

#ifndef NEURAL_NET_H
#define NEURAL_NET_H

// Function pointer representing an element-wise activation function.    [WON'T WORK FOR SOFTMAX]
typedef double (*activation_func)(double);

/*
Perform a feed-forward dense neural network computation serially given relevant parameters.
This function assumes that all vector/matrix dimensionality is correct. 
    e.g. weight matrices should have the correct dimensionality for their previous and current layer sizes.

Parameters:
    `features`: input feature vector (input)
    `num_features`: length of `features` (input)
    `layers`: array of layer sizes where the size of the array is the number of layers excluding the input vector (input)
                e.g. If `layers[2]` is 5, then the THIRD layer after input has FIVE neurons.
    `num_layers`: number of layers in the network excluding an input vector; also the length of `layers` (input)
    `max_layer_size`: largest element of `layers` (input)
    `activation_fns`: An array of function pointers to activation functions. Each function applies to its respective layer,
                    so `activation_fns` is also length `num_layers`. (input)
                        e.g. If `activation_fns[3]` points to a sigmoid function, then layer FOUR will use sigmoid on its neurons.
    `weights`: An array of flattened weight matrices where each matrix applies to its respective layer. So `weights` is also length `num_layers`. (input)
    `biases`: An array of bias vectors where each vector applies to its respective layer, i.e. `biases` is also length `num_layers`. (input)
                e.g. If `biases[2][6]` is 3.4, then the SEVENTH neuron of layer THREE will use bias 3.4.
    `output`: Pointer to an array to hold the final output of the neural network. `*output` should be length `layers[num_layers - 1]`. (output) 

Returns -1 on error, otherwise 0.
*/
int cpu_feed_forward(const double *features,
    const unsigned int num_features,  
    const unsigned int *layers, 
    const unsigned int num_layers,
    const unsigned int max_layer_size,
    const activation_func *activation_fns,
    const double **weights, 
    const double **biases,
    double **output
);

#endif