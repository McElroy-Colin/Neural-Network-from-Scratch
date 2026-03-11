
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda_runtime.h>

#include "memory_utils.h"
#include "neural_net.h"


__global__ void compute_feed_forward(const double *features,
    const unsigned int num_features,  
    const unsigned int *layers, 
    const unsigned int num_layers,
    const unsigned int max_layer_size,
    const ActivationFunc *activation_fns,
    const double **weights, 
    const double **biases,
    double **output,
    int *status) {

}

int nn_load_gpu(const unsigned int num_features,
    const unsigned int *layers,
    const unsigned int num_layers,
    const ActivationFunc *activation_fns,
    const double **weights,
    const double **biases,
    double ***gpu_weights,
    double ***gpu_biases
) {
    // TODO: This can potentially be moved outside of the function call?
    if (!layers || !activation_fns || !weights || !(*weights) || !biases || !(*biases)) {
        fprintf(stderr, "from nn_load_gpu(), uninitialized input pointers\n");
        return -1;
    }

    // Assume the entire network can fit on GPU memory (for now)...
    // Copy features, structure, weights, and biases to the GPU.
    
    cudaError_t err = cudaMalloc(gpu_weights, num_layers*sizeof(double *));
    if (err != cudaSuccess) {
        fprintf(stderr, "from nn_load_gpu(), cuda allocation error");
        return -1;
    }

    err = cudaMalloc(gpu_biases, num_layers*sizeof(double *));
    if (err != cudaSuccess) {
        fprintf(stderr, "from nn_load_gpu(), cuda allocation error");
        cudaFree(gpu_weights);
        return -1;
    }

    int weight_rows = num_features;
    for (int i = 0; i < num_layers; i++) {
        // Allocate this layer's weight matrix and bias to GPU memory.
        double *curr_weights;
        err = cudaMalloc(&curr_weights, weight_rows*layers[i]*sizeof(double));
        if (err != cudaSuccess) {
            fprintf(stderr, "from nn_load_gpu(), cuda allocation error");
            cudafree_2d_darr(*gpu_weights, i);
            cudafree_2d_darr(*gpu_biases, i);
            return -1;
        }
        
        double *curr_biases;
        err = cudaMalloc(&curr_biases, layers[i]*sizeof(double));
        if (err != cudaSuccess) {
            fprintf(stderr, "from nn_load_gpu(), cuda allocation error");
            cudaFree(curr_weights);
            cudafree_2d_darr(*gpu_weights, i);
            cudafree_2d_darr(*gpu_biases, i);
            return -1;
        }

        // Copy this layer's matrix and bias to the allocated GPU memory.
        err = cudaMemcpy(curr_weights, weights[i], weight_rows*layers[i]*sizeof(double), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            fprintf(stderr, "from nn_load_gpu(), cuda memory copy error");
            cudafree_ptrs(curr_weights, curr_biases, NULL);
            cudafree_2d_darr(*gpu_weights, i);
            cudafree_2d_darr(*gpu_biases, i);
            return -1;
        }

        err = cudaMemcpy(curr_biases, biases[i], layers[i]*sizeof(double), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            fprintf(stderr, "from nn_load_gpu(), cuda memory copy error");
            cudafree_ptrs(curr_weights, curr_biases, NULL);
            cudafree_2d_darr(*gpu_weights, i);
            cudafree_2d_darr(*gpu_biases, i);
            return -1;
        }

        // Copy the flattened matrix pointer and bias to the weights array on the GPU.
        err = cudaMemcpy(*gpu_weights + i, &curr_weights, sizeof(double *), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            fprintf(stderr, "from nn_load_gpu(), cuda memory copy error");
            cudafree_ptrs(curr_weights, curr_biases, NULL);
            cudafree_2d_darr(*gpu_weights, i);
            cudafree_2d_darr(*gpu_biases, i);
            return -1;
        }

        err = cudaMemcpy(*gpu_biases + i, &curr_biases, sizeof(double *), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            fprintf(stderr, "from nn_load_gpu(), cuda memory copy error");
            cudafree_ptrs(curr_weights, curr_biases, NULL);
            cudafree_2d_darr(*gpu_weights, i);
            cudafree_2d_darr(*gpu_biases, i);
            return -1;
        }

        weight_rows = layers[i];
    }

    return 0;
}
