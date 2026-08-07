// Source file for the primary neural network functions.

#include <stdio.h>
#include <stdlib.h>
//#include <string.h>

#include "neural_net.h"
#include "train.h"
#include "activations.h"
#include "compute_utils.h"
#include "memory_utils.h"

int nn_dimcheck(unsigned int num_features,
    unsigned int *layers,
    unsigned int num_layers,
    unsigned int num_weights,
    unsigned int num_biases
) {
    if (num_features == 0) {
        fprintf(stderr, "from nn_dimcheck(), num_features must be > 0\n");
        return -1;
    } else if (num_layers == 0) {
        fprintf(stderr, "from nn_dimcheck(), num_layers must be > 0\n");
        return -1;
    } else if (num_weights == 0) {
        fprintf(stderr, "from nn_dimcheck(), num_weights must be > 0\n");
        return -1;
    } else if (num_biases == 0) {
        fprintf(stderr, "from nn_dimcheck(), num_biases must be > 0\n");
        return -1;
    }

    // weight/bias dimensions depend on the layers, including input.
    unsigned int pred_num_weights = layers[0]*num_features;
    unsigned int pred_num_biases = layers[0];
    for (int i = 1; i < num_layers; i++) {
        pred_num_weights += layers[i]*layers[i - 1];
        pred_num_biases += layers[i];
    }

    if (num_weights != pred_num_weights) {
        fprintf(stderr, "from nn_dimcheck(), num_weights does not match layer/feature dimensions\n");
        return -1;
    } else if (num_biases != pred_num_biases) {
        fprintf(stderr, "from nn_dimcheck(), num_biases does not match layer/feature dimensions\n");
        return -1;
    }

    return 0;
}

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
) {
    int err = nn_dimcheck(num_features, layers, num_layers, num_weights, num_biases);
    if (err == -1) {
        fprintf(stderr, "from init_neural_net(), nn_dimcheck() error\n");
        return -1;
    }

    // Calculate maximum layer size including input layer.
    unsigned int max_layer_size = arr_max(layers, num_layers);
    max_layer_size = max_layer_size > num_features ? max_layer_size : num_features;

    // Compute weight and bias offsets.
    unsigned int *weight_offsets = compute_weight_offsets(num_layers, num_features, layers);
    if (!weight_offsets) {
        fprintf(stderr, "from init_neural_net(), memory allocation error\n");
        return -1;
    }
    unsigned int *bias_offsets = compute_bias_offsets(num_layers, layers);
    if (!bias_offsets) {
        fprintf(stderr, "from init_neural_net(), memory allocation error\n");
        free(weight_offsets);
        return -1;
    }

    // Fully initialize the object.
    output_nn->layers = layers;
    output_nn->weights = weights;
    output_nn->biases = biases;
    output_nn->weight_offsets = weight_offsets;
    output_nn->bias_offsets = bias_offsets;
    output_nn->activation_fns = activation_fns;
    output_nn->num_features = num_features;
    output_nn->num_layers = num_layers;
    output_nn->max_layer_size = max_layer_size;
    output_nn->num_weights = num_weights;
    output_nn->num_biases = num_biases;

    return 0;
}

void free_neural_net(NeuralNetwork *neural_net) {
    /* TODO, current test file uses static arrays for some elements of the neural network object.
    free_ptrs(
        neural_net->layers, 
        neural_net->weights,
        neural_net->biases,
        neural_net->weight_offsets,
        neural_net->bias_offsets,
        neural_net->activation_fns,
        NULL
    );*/

    free_ptrs(
        neural_net->weight_offsets,
        neural_net->bias_offsets,
        NULL
    );
}

void feed_forward_hst( 
    const NeuralNetwork *neural_net, // could make it `const NeuralNetwork *restrict neural_net`
    double *y_hats, double *buffer2
) {
    // Hoist pointers from the neural network object to avoid constant dereferencing.
    const double *weights = neural_net->weights;
    const double *biases = neural_net->biases;
    const unsigned int *weight_offsets = neural_net->weight_offsets;
    const unsigned int *bias_offsets = neural_net->bias_offsets;
    const unsigned int *layers = neural_net->layers;
    const ActivationFunc *activation_fns = neural_net->activation_fns;
    const unsigned int num_layers = neural_net->num_layers;
    unsigned int curr_neurons_in = neural_net->num_features;

    // Outer loop goes through one layer at a time, updating buffers with each neuron's activated output.
    for (int l = 0; l < num_layers; l++) {
        // Store local variables for each layer to allow compiler caching on inner iterations.
        const ActivationFunc curr_activation = activation_fns[l];
        const unsigned int curr_neurons_out = layers[l];
        const unsigned int curr_weight_offset = weight_offsets[l];
        const unsigned int curr_bias_offset = bias_offsets[l];

        unsigned int curr_row_index = 0;

        // Inner loop goes through each neuron in the current layer. 
        for (int n = 0; n < curr_neurons_out; n++) {
            double z = 0.0;

            // Innermost loop goes through output from each neuron in the previous layer and computes intermediate weighted sum.
            for (int i = 0; i < curr_neurons_in; i++) {
                z += weights[curr_weight_offset + curr_row_index + i]*y_hats[i];
            }
            z += biases[curr_bias_offset + n];

            curr_row_index += curr_neurons_in;
            // Pass the current neuron's weighted sum to the current layer's activation function.
            buffer2[n] = activation_func_hst(z, curr_activation);
        }

        // After each layer, swap the input and output buffers so that the next layer's input is the current layer's output.
        // Output buffer2 must also be swapped to avoid writing over input data for the next layer.
        double *temp = y_hats;
        y_hats = buffer2;
        buffer2 = temp;
        
        curr_neurons_in = curr_neurons_out;
    }
    // TODO: Wrap the output buffer2, since it is only length layers[num_layers - 1] and y_hats is length largest_layer_size.
    //       Better to do this outside the function?

    return;
}

