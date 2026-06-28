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


double mean_squared_error(const double *y, const double *y_hat, const unsigned int count) {
        double sum = 0.0;
        for (int i = 0; i < count; i++) {
            double diff = y_hat[i] - y[i];
            sum += diff*diff;
        }

        return sum / count;
    }

void batch_mse(const double *ys, 
    const double *y_hats, 
    const unsigned int count_per_vec,
    const unsigned int num_vecs,
    double *errs) {
    
    for (int i = 0; i < num_vecs; i++) {
        errs[i] = mean_squared_error(ys, y_hats, count_per_vec);
    }

    return;
}


int train(const unsigned int num_features,
    const double *features,
    const unsigned int batch_size
    // ...
) {
    /* 
    Steps: 
        Test matrix has a corresponding matrix of output vectors. Take a portion of these pairs as
        test elements and the rest as training.
        Send a vector through feed forward, 
    */

    // Ceiling division
    const unsigned int num_batches = (num_features + batch_size - 1) / batch_size;
    
    // Buffer to store outputs of a batch's forward pass.
    double *y_hats = malloc(batch_size*sizeof(double)); // TODO: Include final layer size here...

    // Outer loop sends batches of feature vectors.
    for (int b = 0; b < num_batches; b++) {
        
        // Forward pass loop for the current batch.
        for (int fp = 0; fp < batch_size; fp++) {
            // ...
        }
    }





    free(y_hats);
}

