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

    const unsigned int num_weights = 7*5 + 4*7 + 2*4;
    double weights[num_weights] = {

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

    const unsigned int num_biases = 7 + 4 + 2;
    double biases[num_biases] = {
        2.4, 43.6, 32.34, 8.454, 45.352, 3.54, 3.561,

        2.4, 43.6, 32.34, 8.454,

        3.54, 3.561
    };

    double **feature_matrix;
    unsigned int *feature_lengths;
    
    // Convert the given CSV file to a matrix of doubles, so each row represents a feature vector.
    const int num_feature_vectors = csv_to_matrix("sample_data/floats.csv", 1, 30, 10, &feature_matrix, &feature_lengths);
    if (num_feature_vectors == -1) {
        fprintf(stderr, "from main(), csv_to_matrix() error\n");
        return 1;
    }

    // Assume all feature vectors are the same length.
    const unsigned int num_features = feature_lengths[0];

    unsigned int num_layers = 3;
    unsigned int layers[num_layers] = {7, 4, 2};

    ActivationFunc activation_fns[num_layers] = {RELU, RELU, RELU};

    NeuralNetwork host_nn;
    int err = init_neural_net(
        num_features,
        layers,
        num_layers,
        weights,
        num_weights,
        biases,
        num_biases,
        activation_fns,
        &host_nn
    );

    if (err == -1) {
        fprintf(stderr, "from main(), init_neural_net() error\n");
        free(feature_lengths);
        free_2d_darr(feature_matrix, num_feature_vectors);
        return 1;
    }

    double *buffer1, *buffer2;
    NeuralNetwork shell_nn;
    err = nn_load_dvc(&host_nn, &shell_nn, &buffer1, &buffer2);
    if (err == -1) {
        fprintf(stderr, "from main(), nn_load_dvc() error\n");
        free_ptrs(feature_lengths, host_nn.weight_offsets, host_nn.bias_offsets, NULL);
        free_2d_darr(feature_matrix, num_feature_vectors);
        return 1;
    }

    dim3 num_blocks((shell_nn.max_layer_size + TILE_SIZE - 1) / TILE_SIZE);
    dim3 threads_per_block(TILE_SIZE, TILE_SIZE);

    double *output = (double*)malloc(2*sizeof(double));
    if (!output) {
         fprintf(stderr, "from main(), memory allocation error error\n");
        free_ptrs(feature_lengths, host_nn.weight_offsets, host_nn.bias_offsets, NULL);
        free_2d_darr(feature_matrix, num_feature_vectors);
        cudafree_neural_net(&shell_nn);
        return 1;
    }

    for (int i = 0; i < num_feature_vectors; i++) {
        cudaMemcpy(buffer1, feature_matrix[i], 5*sizeof(double), cudaMemcpyHostToDevice);

        feed_forward_dvc<<<num_blocks, threads_per_block>>>(
            shell_nn,
            buffer1, buffer2
        );

        cudaMemcpy(output, buffer1, 2*sizeof(double), cudaMemcpyDeviceToHost);
        printf("Output f%d: (%f, %f)\n", i + 1, output[0], output[1]);
    }

    cudafree_neural_net(&shell_nn);
    free_neural_net(&host_nn);
    cudafree_ptrs(buffer1, buffer2, NULL);
    free_2d_darr(feature_matrix, num_feature_vectors);
    return 0;
}