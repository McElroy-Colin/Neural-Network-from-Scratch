// Source file for the primary neural network functions.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "neural_net.h"
#include "activation_functions.h"


int feed_forward_srl(
    const double *features, 
    const unsigned int num_features, 
    const unsigned int *layers, 
    const unsigned int num_layers,
    const unsigned int max_layer_size,
    const ActivationFunc *activation_fns,
    const double **weights,
    const double **biases,
    double **output
) {
    // TODO: This can potentially be moved outside of the function call?
    if (!features || !layers || !activation_fns || !weights || !(*weights) || !biases || !(*biases)) {
        fprintf(stderr, "from feed_forward_srl(), uninitialized input pointers\n");
        return -1;
    }

    // Allocate two buffers, each able to hold any given layer at a time.
    // TODO: Move this outside of the function call!
    const unsigned int largest_layer_size = (max_layer_size > num_features) ? max_layer_size : num_features;
    double *layer_input = malloc(largest_layer_size*sizeof(double));
    if (!layer_input) {
        fprintf(stderr, "from feed_forward_srl(), allocation error");
        return -1;
    }
    double *layer_output = malloc(largest_layer_size*sizeof(double));
    if (!layer_output) {
        fprintf(stderr, "from feed_forward_srl(), allocation error");
        free(layer_input);
        return -1;
    }

    unsigned int curr_neurons_in = num_features;
    memcpy(layer_input, features, num_features*sizeof(double));

    // Outer loop goes through one layer at a time, updating buffers with each neuron's activated output.
    for (int l = 0; l < num_layers; l++) {
        const double *curr_weights = weights[l];
        const double *curr_biases = biases[l];
        ActivationFunc curr_activation = activation_fns[l];
        unsigned int curr_neurons_out = layers[l];
        unsigned int curr_row = 0;

        // Inner loop goes through each neuron in the current layer. 
        for (int n = 0; n < curr_neurons_out; n++) {
            double z = 0.0;

            // Innermost loop goes through output from each neuron in the previous layer and computes intermediate weighted sum.
            for (int i = 0; i < curr_neurons_in; i++) {
                z += curr_weights[curr_row + i]*layer_input[i];
            }
            z += curr_biases[n];

            curr_row += curr_neurons_in;
            // Pass the current neuron's weighted sum to the current layer's activation function.
            layer_output[n] = activation_func_srl(z, curr_activation);
        }

        // After each layer, swap the input and output buffers so that the next layer's input is the current layer's output.
        // Output buffer must also be swapped to avoid writing over input data for the next layer.
        double *temp = layer_input;
        layer_input = layer_output;
        layer_output = temp;
        
        curr_neurons_in = curr_neurons_out;
    }

    // Since we swap buffers after each layer, the layer_input pointer actually points to the output of the final layer.
    *output = layer_input;

    // TODO: Wrap the output buffer, since it is only length layers[num_layers - 1] and layer_input is length largest_layer_size

    free(layer_output);
    return 0;
}