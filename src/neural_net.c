// Source file for the primary neural network functions.

#include <stdio.h>
#include <stdlib.h>
//#include <string.h>

#include "neural_net.h"
#include "activation_functions.h"
#include "compute_utils.h"
#include "memory_utils.h"


// TODO return by value here
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
    /* TEMP, current test file uses static arrays for some elements of the neural network object.
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
    double *buffer1, double *buffer2
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
                z += weights[curr_weight_offset + curr_row_index + i]*buffer1[i];
            }
            z += biases[curr_bias_offset + n];

            curr_row_index += curr_neurons_in;
            // Pass the current neuron's weighted sum to the current layer's activation function.
            buffer2[n] = activation_func_hst(z, curr_activation);
        }

        // After each layer, swap the input and output buffers so that the next layer's input is the current layer's output.
        // Output buffer must also be swapped to avoid writing over input data for the next layer.
        double *temp = buffer1;
        buffer1 = buffer2;
        buffer2 = temp;
        
        curr_neurons_in = curr_neurons_out;
    }
    // TODO: Wrap the output buffer, since it is only length layers[num_layers - 1] and buffer1 is length largest_layer_size.
    //       Better to do this outside the function?

    return;
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


int train_hst(
    NeuralNetwork *neural_net,
    const double *features, // Flattened array of feature vectors
    const unsigned int batch_size
    // ...
) {
    // Hoist pointers from the neural network object to avoid constant dereferencing.
    double *weights = neural_net->weights;
    double *biases = neural_net->biases;
    const unsigned int *weight_offsets = neural_net->weight_offsets;
    const unsigned int *bias_offsets = neural_net->bias_offsets;
    const unsigned int *layers = neural_net->layers;
    const ActivationFunc *activation_fns = neural_net->activation_fns;
    const unsigned int num_features = neural_net->num_features;
    const unsigned int num_layers = neural_net->num_layers;
    const unsigned int max_layer_size = neural_net->max_layer_size;
    /* 
    Steps: 
        Test matrix has a corresponding matrix of output vectors. Take a portion of these pairs as
        test elements and the rest as training.
        Send a vector through feed forward, get loss, do gradient for each different act func, backprop.
    */

    // Ceiling division
    const unsigned int num_batches = (num_features + batch_size - 1) / batch_size;
    
    // Buffer to store outputs of a batch's forward pass.
    double *y_hats = malloc(batch_size*layers[num_layers - 1]*sizeof(double));
    if (!y_hats) {
        fprintf(stderr, "from train_hst(), memory allocation error\n");
        return -1;
    }
    double *buffer1 = malloc(max_layer_size*sizeof(double));
    if (!buffer1) {
        fprintf(stderr, "from train_hst(), memory allocation error\n");
        free(y_hats);
        return -1;
    }
    double *buffer2 = malloc (max_layer_size*sizeof(double));
    if (!buffer2) {
        fprintf(stderr, "from train_hst(), memory allocation error\n");
        free_ptrs(y_hats, buffer1, NULL);
        return -1;
    }

    // Outer loop sends batches of feature vectors.
    for (int b = 0; b < num_batches; b++) {
        
        // Forward pass loop for the current batch.
        for (int fp = 0; fp < batch_size; fp++) {
            // Store buffer1 (output) in y_hats for each forward pass of the batch...
            //feed_forward_hst();
        }
    }





    free(y_hats);
}

