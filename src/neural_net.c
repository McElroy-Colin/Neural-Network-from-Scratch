// Source file for the primary neural network functions.

#include <stdio.h>
#include <stdlib.h>
//#include <string.h>

#include "neural_net.h"
#include "activation_functions.h"


int feed_forward_hst( 
    const unsigned int num_features, 
    const unsigned int *layers, 
    const unsigned int num_layers,
    const unsigned int *layer_offsets,
    const ActivationFunc *activation_fns,
    const double *weights,
    const double *biases,
    double *buffer1, double *buffer2
) {
    unsigned int curr_neurons_in = num_features;
    unsigned int total_neurons = 0;

    // Outer loop goes through one layer at a time, updating buffers with each neuron's activated output.
    for (int l = 0; l < num_layers; l++) {
        ActivationFunc curr_activation = activation_fns[l];
        unsigned int curr_neurons_out = layers[l];
        unsigned int curr_layer_offset = layer_offsets[l];
        unsigned int curr_row = 0;

        // Inner loop goes through each neuron in the current layer. 
        for (int n = 0; n < curr_neurons_out; n++) {
            double z = 0.0;

            // Innermost loop goes through output from each neuron in the previous layer and computes intermediate weighted sum.
            for (int i = 0; i < curr_neurons_in; i++) {
                z += weights[curr_layer_offset + curr_row + i]*buffer1[i];
            }
            z += biases[total_neurons + n];

            curr_row += curr_neurons_in;
            // Pass the current neuron's weighted sum to the current layer's activation function.
            buffer2[n] = activation_func_hst(z, curr_activation);
        }

        total_neurons += curr_neurons_out;

        // After each layer, swap the input and output buffers so that the next layer's input is the current layer's output.
        // Output buffer must also be swapped to avoid writing over input data for the next layer.
        double *temp = buffer1;
        buffer1 = buffer2;
        buffer2 = temp;
        
        curr_neurons_in = curr_neurons_out;
    }
    // TODO: Wrap the output buffer, since it is only length layers[num_layers - 1] and buffer1 is length largest_layer_size.
    //       Better to do this outside the function.

    return 0;
}


int train(const unsigned int num_features,  
    const unsigned int *layers, 
    const unsigned int num_layers,
    const unsigned int *layer_offsets,
    const ActivationFunc *activation_fns,
    const double *weights, 
    const double *biases,
    double *buffer1, double *buffer2,
    const double *train_matrix, 
    const double *test_matrix
) {
    // Steps: 
    
}

