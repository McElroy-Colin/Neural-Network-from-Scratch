#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#include "memory_utils.cuh"

extern "C" {
#include "text_utils.h"
#include "memory_utils.h"
#include "compute_utils.h"
}

#include "constants.h"

#include "neural_net.cuh"
#include "activation_functions.cuh"


// The results are accurate with tile size smaller than that of `mul_vector` and otherwise.
int main(int argc, char** argv) {
    // TEST: 5 (in) -> 7 -> 4 -> 2 (out)

    double weights[7*5 + 4*7 + 2*4] = {

        // input layer -> 1 7x5

        2.4, 43.6, 32.34, 8.454, 45.352,
        2.4, 44, 32.34, 8.5, 45.352,
        2.4, 43.6, 32.34, 8.454, 45.352,
        2.4, 43.6, 32.34, 8.454, 45.352,
        2.4, 43.6, 32.34, 8.454, 45.352,
        2.4, 43.6, 32.34, 8.454, 45.352,
        2.4, 43.6, 32.34, 8.454, 45.352,

        // layer 1 -> 2 4x7

        2.4, 43.6, 32.34, 8.454, 45.352, 3.24, 4.213,
        2.4, 43.6, 32.34, 8.454, 45.352, 3.24, 4.213,
        2.4, 43.6, 32.34, 8.454, 45.352, 3.24, 4.213,
        2.4, 43.6, 32.34, 8.45, 45.352, 3.24, 4.213,

        // layer 2 -> 3 2x4

        2.4, 43.6, 32.34, 8.454,
        2.4, 43.6, 32.34, 8.454
    };

    double biases[7 + 4 + 2] = {
        2.4, 43.6, 32.34, 8.454, 45.352, 3.54, 3.561,

        2.4, 43.6, 32.34, 8.454,

        3.54, 3.561
    };

    double **feature_matrix;
    unsigned int *feature_lengths;
    
    // Convert the given CSV file to a matrix of doubles, so each row represents a feature vector.
    const int num_feature_vectors = csv_to_matrix("sample_data/floats.csv", 1, 30, 10, &feature_matrix, &feature_lengths);
    if (num_feature_vectors == -1) {
        fprintf(stderr, "from main(), csv_to_matrix() error");
        return 1;
    }

    unsigned int layers[3] = {7, 4, 2};
    const unsigned int max_layer = 7;

    unsigned int *layer_offsets = (unsigned int*)malloc(5*sizeof(unsigned int));
    compute_layer_offsets(3, 5, layers, layer_offsets);

    ActivationFunc activation_fns[3] = {RELU, RELU, RELU};

    NeuralNetwork host_nn = {
        .layers = layers,
        .weights = weights,
        .biases = biases,
        .weight_offsets = layer_offsets,
        .bias_offsets = 0, // TODO
        .activation_fns = activation_fns,
        .num_features = num_feature_vectors,
        .num_layers = 3,
        .max_layer_size = max_layer,
        .total_weights = 7*5 + 4*7 + 2*4,
        .total_biases = 7 + 4 + 2
    };

    unsigned int *dvc_layers, *dvc_layer_offsets;
    double *dvc_weights, *dvc_biases, *buffer1, *buffer2;
    ActivationFunc *dvc_activation_fns;

    NeuralNetwork shell_nn;

    nn_load_dvc(&host_nn, &shell_nn, &buffer1, &buffer2);

    dim3 num_blocks((max_layer + TILE_SIZE - 1) / TILE_SIZE);
    dim3 threads_per_block(TILE_SIZE, TILE_SIZE);

    double *output = (double*)malloc(2*sizeof(double));

    for (int i = 0; i < num_feature_vectors; i++) {
        cudaMemcpy(buffer1, feature_matrix[i], 5*sizeof(double), cudaMemcpyHostToDevice);

        feed_forward_dvc<<<num_blocks, threads_per_block>>>(
            5,
            dvc_layers,
            3,
            dvc_layer_offsets,
            dvc_activation_fns,
            dvc_weights,
            dvc_biases,
            buffer1, buffer2
        );

        cudaMemcpy(output, buffer1, 2*sizeof(double), cudaMemcpyDeviceToHost);
        printf("Output f%d: (%f, %f)\n", i + 1, output[0], output[1]);
    }

    cudafree_ptrs(dvc_weights, dvc_layer_offsets, dvc_biases, dvc_activation_fns, buffer1, buffer2, NULL);
    free_2d_darr(feature_matrix, num_feature_vectors);
    free_ptrs(output, layer_offsets, NULL);
    return 0;
}