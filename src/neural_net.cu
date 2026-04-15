
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda_runtime.h>

#include "memory_utils.cuh"
#include "neural_net.cuh"
#include "activation_functions.cuh"
#include "compute_utils.h"
#include "constants.h"


__global__ void feed_forward_dvc(const unsigned int num_features,  
    const unsigned int *layers, 
    const unsigned int num_layers,
    const unsigned int *layer_offsets,
    const ActivationFunc *activation_fns,
    const double *weights, 
    const double *biases,
    double *buffer1, double *buffer2
) {
    // Designate a tile of shared memory for each active vector.
    __shared__ double tile[TILE_SIZE];

    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int curr_neurons_in = num_features;
    unsigned int total_neurons = 0;

    for (int l = 0; l < num_layers; l++) {
        ActivationFunc curr_activation = activation_fns[l];
        unsigned int curr_neurons_out = layers[l];
        unsigned int curr_layer_offset = layer_offsets[l];
        double z = 0.0;

        // Loop over tiles of the input vector.
        for (unsigned int t = 0; t < (curr_neurons_in + TILE_SIZE - 1)/TILE_SIZE; ++t) {
            unsigned int curr_chunk = t*TILE_SIZE;

            // Load a tile of the current buffer into shared memory.
            unsigned int curr_x_index = curr_chunk + threadIdx.y;
            if ((curr_x_index < curr_neurons_in) && (threadIdx.x == 0)) {
                tile[threadIdx.y] = buffer1[curr_x_index];
            }

            __syncthreads(); // Ensure tile is loaded

            // Perform a feed forwad computation.
            if (idx < curr_neurons_out) {
                for (unsigned int i = 0; i < TILE_SIZE; ++i) {
                    unsigned int c = curr_chunk + i;
                    if (c < curr_neurons_in) {
                        z += weights[curr_layer_offset + idx*curr_neurons_in + c]*tile[i];
                    }
                }
            }

            __syncthreads(); // Ensure all threads are done using shared memory
        }

        // Add the biases and pass through the activation functions.
        if (idx < curr_neurons_out) {
            buffer2[idx] = activation_func_dvc(z + biases[total_neurons + idx], curr_activation);
        }

        total_neurons += curr_neurons_out;

        // Exchange the computed layer for the previous layer variables.
        curr_neurons_in = curr_neurons_out;
        double *temp = buffer1;
        buffer1 = buffer2;
        buffer2 = temp;
    }

    // buffer1 holds the result on the device.
    return;
}

int nn_load_dvc(const unsigned int *layers,
    const unsigned int num_layers,
    const unsigned int num_features,
    const unsigned int *layer_offsets,
    const ActivationFunc *activation_fns,
    const double *weights,
    const unsigned int num_weights,
    const double *biases,
    const unsigned int num_biases,
    unsigned int **dvc_layers,
    double **dvc_weights,
    unsigned int **dvc_layer_offsets,
    double **dvc_biases,
    ActivationFunc **dvc_activation_fns,
    double **dvc_buffer1, double **dvc_buffer2
) {
    // Assume the entire network can fit on device memory (for now)...
    // Allocate and copy structure, weights, biases, and buffers to the device.

    cudaError_t err = cudaMalloc(dvc_layers, num_layers*sizeof(unsigned int));
    if (err != cudaSuccess) {
        fprintf(stderr, "from nn_load_dvc(), cuda allocation error");
        return -1;
    }

    err = cudaMemcpy(*dvc_layers, layers, num_layers*sizeof(unsigned int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "from nn_load_dvc(), cuda memory copy error");
        cudaFree(dvc_layers);
        return -1;
    }

    err = cudaMalloc(dvc_weights, num_weights*sizeof(double));
    if (err != cudaSuccess) {
        fprintf(stderr, "from nn_load_dvc(), cuda allocation error");
        cudaFree(dvc_layers);
        return -1;
    }

    err = cudaMemcpy(*dvc_weights, weights, num_weights*sizeof(double), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "from nn_load_dvc(), cuda memory copy error");
        cudafree_ptrs(*dvc_layers, *dvc_weights, NULL);
        return -1;
    }

    err = cudaMalloc(dvc_layer_offsets, num_layers*sizeof(unsigned int));
    if(err != cudaSuccess) {
        fprintf(stderr, "from nn_load_dvc(), cuda allocation error");
        cudafree_ptrs(*dvc_layers, *dvc_weights, NULL);
        return -1;
    }

    err = cudaMemcpy(*dvc_layer_offsets, layer_offsets, num_layers*sizeof(unsigned int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "from nn_load_dvc(), cuda memory copy error");
        cudafree_ptrs(*dvc_layers, *dvc_weights, *dvc_layer_offsets, NULL);
        return -1;
    }

    err = cudaMalloc(dvc_biases, num_biases*sizeof(double));
    if (err != cudaSuccess) {
        fprintf(stderr, "from nn_load_dvc(), cuda allocation error");
        cudafree_ptrs(*dvc_layers, *dvc_weights, *dvc_layer_offsets, NULL);
        return -1;
    }

    err = cudaMemcpy(*dvc_biases, biases, num_biases*sizeof(double), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "from nn_load_dvc(), cuda memory copy error");
        cudafree_ptrs(*dvc_layers, *dvc_weights, *dvc_layer_offsets, *dvc_biases, NULL);
        return -1;
    }

    err = cudaMalloc(dvc_activation_fns, num_layers*sizeof(ActivationFunc));
    if (err != cudaSuccess) {
        fprintf(stderr, "from nn_load_dvc(), cuda allocation error");
        cudafree_ptrs(*dvc_layers, *dvc_weights, *dvc_layer_offsets, *dvc_biases, NULL);
        return -1;
    }

    err = cudaMemcpy(*dvc_activation_fns, activation_fns, num_layers*sizeof(ActivationFunc), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "from nn_load_dvc(), cuda memory copy error");
        cudafree_ptrs(*dvc_layers, *dvc_weights, *dvc_layer_offsets, *dvc_biases, *dvc_activation_fns, NULL);
        return -1;
    }

    // Compute the largest layer size, including the input (feature) layer.
    const unsigned int max_layer = max(arr_max(layers, num_layers), num_features);

    // Allocate the two necessary buffers on the device.

    err = cudaMalloc(dvc_buffer1, max_layer*sizeof(double));
    if (err != cudaSuccess) {
        fprintf(stderr, "from nn_load_dvc(), cuda allocation error");
        cudafree_ptrs(*dvc_layers, *dvc_weights, *dvc_layer_offsets, *dvc_biases, *dvc_activation_fns, NULL);
        return -1;
    }

    err = cudaMalloc(dvc_buffer2, max_layer*sizeof(double));
    if (err != cudaSuccess) {
        fprintf(stderr, "from nn_load_dvc(), cuda allocation error");
        cudafree_ptrs(*dvc_layers, *dvc_weights, *dvc_layer_offsets, *dvc_biases, *dvc_activation_fns, NULL);
        return -1;
    }

    // Return with the gpu buffers assigned.

    return 0;
}
