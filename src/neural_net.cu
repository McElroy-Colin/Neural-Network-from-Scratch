
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda_runtime.h>

#include "memory_utils.h"
#include "neural_net.h"
#include "compute_utils.h"
#include "constants.h"



// assume buffer1 holds the feature vector...
__global__ void feed_forward_dvc(const unsigned int num_features,  
    const unsigned int *layers, 
    const unsigned int num_layers,
    const unsigned int *layer_offsets,
    const unsigned int max_layer_size,
    const ActivationFunc *activation_fns,
    const double *weights, 
    const double *biases,
    double *buffer1, double *buffer2
) {
    // Designate a tile of shared memory for each active vector.
    __shared__ double tile[TILE_SIZE];

    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int curr_neurons_in = num_features;
    unsigned int total_neurons = curr_neurons_in;

    for (int l = 0; l < num_layers; l++) {
        ActivationFunc curr_activation = activation_fns[l];
        unsigned int curr_neurons_out = layers[l];
        unsigned int curr_layer_offset = layer_offsets[l];
        double sum = 0.0;

        total_neurons += curr_neurons_out;

        // Loop over tiles of the input vector
        for (unsigned int t = 0; t < (curr_neurons_in + TILE_SIZE - 1)/TILE_SIZE; ++t) {
            // Load a tile of the current buffer into shared memory
            unsigned int curr_x_index = t*TILE_SIZE + threadIdx.y;
            if ((curr_x_index < curr_neurons_in) && (threadIdx.x == 0)) {
                tile[threadIdx.y] = buffer1[curr_x_index];
            }

            __syncthreads(); // Ensure tile is loaded

            // Perform a feed_forwad computation.
            if (idx < curr_neurons_out) {
                for (unsigned int i = 0; i < TILE_SIZE; ++i) {
                    unsigned int c = t*TILE_SIZE + i;
                    if (c < curr_neurons_in) {
                        sum += weights[layer_offsets[l] + idx*curr_neurons_in + c]*tile[i];
                    }
                }
            }

            __syncthreads(); // Ensure all threads are done using shared memory
        }

        // Add the biases and pass through the activation functions.
        if (idx < curr_neurons_out) {
            buffer2[idx] = activation_func_dvc(sum + biases[total_neurons + idx], curr_activation);
        }

        // Exchange the computed layer for the previous layer variables.
        curr_neurons_in = curr_neurons_out;
        double *temp = buffer1;
        buffer1 = buffer2;
        buffer2 = temp;
    }

    // buffer1 holds the result on the device.
    return;
}

int nn_load_dvc(const unsigned int max_layer,
    const unsigned int num_layers,
    const unsigned int *layer_offsets,
    const ActivationFunc *activation_fns,
    const double *weights,
    const unsigned int num_weights,
    const double *biases,
    const unsigned int num_biases,
    double **dvc_weights,
    double **dvc_layer_offsets,
    double **dvc_biases,
    ActivationFunc **dvc_activation_fns,
    double **dvc_buffer1, double **dvc_buffer2
) {
    // Assume the entire network can fit on device memory (for now)...
    // Copy structure, weights, biases, and buffers to the device.

    cudaError_t err = cudaMalloc(dvc_weights, num_weights*sizeof(double));
    if (err != cudaSuccess) {
        fprintf(stderr, "from nn_load_dvc(), cuda allocation error");
        return -1;
    }

    err = cudaMalloc(dvc_layer_offsets, num_layers*sizeof(unsigned int));
    if(err != cudaSuccess) {
        fprintf(stderr, "from nn_load_dvc(), cuda allocation error");
       cudaFree(*dvc_weights);
        return -1;
    }

    err = cudaMalloc(dvc_biases, num_biases*sizeof(double));
    if (err != cudaSuccess) {
        fprintf(stderr, "from nn_load_dvc(), cuda allocation error");
        cudafree_ptrs(*dvc_weights, *dvc_layer_offsets, NULL);
        return -1;
    }

    err = cudaMalloc(dvc_activation_fns, num_layers*sizeof(ActivationFunc));
    if (err != cudaSuccess) {
        fprintf(stderr, "from nn_load_dvc(), cuda allocation error");
        cudafree_ptrs(*dvc_weights, *dvc_layer_offsets, *dvc_biases, NULL);
        return -1;
    }

    err = cudaMalloc(dvc_buffer1, max_layer*sizeof(double));
    if (err != cudaSuccess) {
        fprintf(stderr, "from nn_load_dvc(), cuda allocation error");
        cudafree_ptrs(*dvc_weights, *dvc_layer_offsets, *dvc_biases, *dvc_activation_fns, NULL);
        return -1;
    }

    err = cudaMalloc(dvc_buffer2, max_layer*sizeof(double));
    if (err != cudaSuccess) {
        fprintf(stderr, "from nn_load_dvc(), cuda allocation error");
        cudafree_ptrs(*dvc_weights, *dvc_layer_offsets, *dvc_biases, *dvc_activation_fns, NULL);
        return -1;
    }

    err = cudaMemcpy(*dvc_weights, weights, num_weights*sizeof(double), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "from nn_load_dvc(), cuda memory copy error");
        cudafree_ptrs(*dvc_weights, *dvc_layer_offsets, *dvc_biases, *dvc_activation_fns, NULL);
        return -1;
    }

    err = cudaMemcpy(*dvc_layer_offsets, layer_offsets, num_layers*sizeof(unsigned int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "from nn_load_dvc(), cuda memory copy error");
        cudafree_ptrs(*dvc_weights, *dvc_layer_offsets, *dvc_biases, *dvc_activation_fns, NULL);
        return -1;
    }


    err = cudaMemcpy(*dvc_biases, biases, num_biases*sizeof(double), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "from nn_load_dvc(), cuda memory copy error");
        cudafree_ptrs(*dvc_weights, *dvc_layer_offsets, *dvc_biases, *dvc_activation_fns, NULL);
        return -1;
    }

    // Return with the gpu buffers assigned.

    return 0;
}
