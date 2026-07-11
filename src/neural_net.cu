
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda_runtime.h>

#include "memory_utils.cuh"
#include "neural_net.cuh"
#include "activation_functions.cuh"
#include "compute_utils.h"
#include "constants.h"
#include "text_utils.h"


__global__ void feed_forward_dvc(
    NeuralNetwork neural_net,
    double *buffer1, double *buffer2
) {
    // Designate a tile of shared memory for each active vector.
    __shared__ double tile[TILE_SIZE];

    unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;
    unsigned int curr_neurons_in = neural_net.num_features;

    unsigned int *layers = neural_net.layers;
    double *weights = neural_net.weights;
    double *biases = neural_net.biases;
    unsigned int *weight_offsets = neural_net.weight_offsets;
    unsigned int *bias_offsets = neural_net.bias_offsets;
    ActivationFunc *activation_fns = neural_net.activation_fns;
    unsigned int num_layers = neural_net.num_layers;
    

    for (int l = 0; l < num_layers; l++) {
        ActivationFunc curr_activation = activation_fns[l];
        unsigned int curr_neurons_out = layers[l];
        unsigned int curr_weight_offset = weight_offsets[l];
        unsigned int curr_bias_offset = bias_offsets[l];
        double z = 0.0;

        // Loop over tiles of the input vector.
        for (unsigned int t = 0; t < (curr_neurons_in + TILE_SIZE - 1)/TILE_SIZE; ++t) {
            unsigned int curr_chunk = t*TILE_SIZE;

            // TODO: reuse x dimension threads to load cooperatively, otherwise y dimension threads are wasted...
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
                        z += weights[curr_weight_offset + idx*curr_neurons_in + c]*tile[i];
                    }
                }
            }

            __syncthreads(); // Ensure all threads are done using shared memory
        }

        // Add the biases and pass through the activation functions.
        if (idx < curr_neurons_out) {
            buffer2[idx] = activation_func_dvc(z + biases[curr_bias_offset + idx], curr_activation); // TODO fix
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

int nn_load_dvc(
    NeuralNetwork *hst_neural_net,
    NeuralNetwork *shell_neural_net,
    double **dvc_buffer1, double **dvc_buffer2
) {
    // Assume the entire network can fit on device memory (for now)...
    // Allocate and copy structure, weights, biases, and buffers to the device.

    // First, copy fixed size data to the host structure.
    shell_neural_net->num_features = hst_neural_net->num_features;
    shell_neural_net->num_layers = hst_neural_net->num_layers;
    shell_neural_net->max_layer_size = hst_neural_net->max_layer_size;
    shell_neural_net->num_weights = hst_neural_net->num_weights;
    shell_neural_net->num_biases = hst_neural_net->num_biases;

    // Now, allocate GPU memory arrays.

    cudaError_t err = cudaMalloc(&shell_neural_net->layers, shell_neural_net->num_layers*sizeof(unsigned int));
    if (err != cudaSuccess) {
        fprintf(stderr, "from nn_load_dvc(), cuda allocation error\n");
        return -1;
    }

    err = cudaMemcpy(shell_neural_net->layers, hst_neural_net->layers, shell_neural_net->num_layers*sizeof(unsigned int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "from nn_load_dvc(), cuda memory copy error1\n");
        cudaFree(shell_neural_net->layers);
        return -1;
    }

    err = cudaMalloc(&shell_neural_net->weights, shell_neural_net->num_weights*sizeof(double));
    if (err != cudaSuccess) {
        fprintf(stderr, "from nn_load_dvc(), cuda allocation error\n");
        cudaFree(shell_neural_net->layers);
        return -1;
    }

    err = cudaMemcpy(shell_neural_net->weights, hst_neural_net->weights, shell_neural_net->num_weights*sizeof(double), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "from nn_load_dvc(), cuda memory copy error2\n");
        cudafree_ptrs(shell_neural_net->layers, shell_neural_net->weights, NULL);
        return -1;
    }

    err = cudaMalloc(&shell_neural_net->weight_offsets, shell_neural_net->num_layers*sizeof(unsigned int));
    if (err != cudaSuccess) {
        fprintf(stderr, "from nn_load_dvc(), cuda allocation error\n");
        cudafree_ptrs(shell_neural_net->layers, shell_neural_net->weights, NULL);
        return -1;
    }

    err = cudaMemcpy(shell_neural_net->weight_offsets, hst_neural_net->weight_offsets, shell_neural_net->num_layers*sizeof(unsigned int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "from nn_load_dvc(), cuda memory copy error3\n");
        cudafree_ptrs(shell_neural_net->layers, shell_neural_net->weights, shell_neural_net->weight_offsets, NULL);
        return -1;
    }

    err = cudaMalloc(&shell_neural_net->biases, shell_neural_net->num_biases*sizeof(double));
    if (err != cudaSuccess) {
        fprintf(stderr, "from nn_load_dvc(), cuda allocation error\n");
        cudafree_ptrs(shell_neural_net->layers, shell_neural_net->weights, shell_neural_net->weight_offsets, NULL);
        return -1;
    }

    err = cudaMemcpy(shell_neural_net->biases, hst_neural_net->biases, shell_neural_net->num_biases*sizeof(double), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "from nn_load_dvc(), cuda memory copy error4\n");
        cudafree_ptrs(shell_neural_net->layers, shell_neural_net->weights, shell_neural_net->weight_offsets, shell_neural_net->biases, NULL);
        return -1;
    }

    err = cudaMalloc(&shell_neural_net->bias_offsets, shell_neural_net->num_layers*sizeof(unsigned int));
    if (err != cudaSuccess) {
        fprintf(stderr, "from nn_load_dvc(), cuda allocation error\n");
        cudafree_ptrs(shell_neural_net->layers, shell_neural_net->weights, shell_neural_net->weight_offsets, shell_neural_net->biases, NULL);
        return -1;
    }

    err = cudaMemcpy(shell_neural_net->bias_offsets, hst_neural_net->bias_offsets, shell_neural_net->num_layers*sizeof(unsigned int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "from nn_load_dvc(), cuda memory copy error5\n");
        cudafree_ptrs(shell_neural_net->layers, 
            shell_neural_net->weights, 
            shell_neural_net->weight_offsets, 
            shell_neural_net->biases, 
            shell_neural_net->bias_offsets, 
            NULL);
        return -1;
    }

    err = cudaMalloc(&shell_neural_net->activation_fns, shell_neural_net->num_layers*sizeof(ActivationFunc));
    if (err != cudaSuccess) {
        fprintf(stderr, "from nn_load_dvc(), cuda allocation error\n");
        cudafree_ptrs(shell_neural_net->layers, 
            shell_neural_net->weights, 
            shell_neural_net->weight_offsets, 
            shell_neural_net->biases, 
            shell_neural_net->bias_offsets, 
            NULL);
        return -1;
    }

    err = cudaMemcpy(shell_neural_net->activation_fns, hst_neural_net->activation_fns, shell_neural_net->num_layers*sizeof(ActivationFunc), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "from nn_load_dvc(), cuda memory copy error6\n");
        cudafree_ptrs(shell_neural_net->layers, 
            shell_neural_net->weights, 
            shell_neural_net->weight_offsets, 
            shell_neural_net->biases, 
            shell_neural_net->bias_offsets, 
            shell_neural_net->activation_fns, 
            NULL);
        return -1;
    }

    // Allocate the two necessary buffers on the device.
    // max_layer_size was already copied from the host struct above.

    err = cudaMalloc(dvc_buffer1, shell_neural_net->max_layer_size*sizeof(double));
    if (err != cudaSuccess) {
        fprintf(stderr, "from nn_load_dvc(), cuda allocation error\n");
        cudafree_ptrs(shell_neural_net->layers, 
            shell_neural_net->weights, 
            shell_neural_net->weight_offsets, 
            shell_neural_net->biases, 
            shell_neural_net->bias_offsets, 
            shell_neural_net->activation_fns, 
            NULL);
        return -1;
    }

    err = cudaMalloc(dvc_buffer2, shell_neural_net->max_layer_size*sizeof(double));
    if (err != cudaSuccess) {
        fprintf(stderr, "from nn_load_dvc(), cuda allocation error\n");
        cudafree_ptrs(shell_neural_net->layers, 
            shell_neural_net->weights, 
            shell_neural_net->weight_offsets, 
            shell_neural_net->biases, 
            shell_neural_net->bias_offsets, 
            shell_neural_net->activation_fns, 
            NULL);
        return -1;
    }

    // Return with the device buffers assigned.

    return 0;
}
