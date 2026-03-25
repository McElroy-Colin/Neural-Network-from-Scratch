
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <cuda_runtime.h>

#include "memory_utils.h"
#include "neural_net.h"
#include "compute_utils.h"


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

int feed_forward_krnl(const double *features,
    const unsigned int num_features,  
    const unsigned int *layers, 
    const unsigned int num_layers,
    const unsigned int *layer_offsets,
    const unsigned int max_layer_size,
    const ActivationFunc *activation_fns,
    const double *gpu_weights, 
    const double *gpu_biases,
    double *buffer1, double *buffer2,
    double **output
) {
    // TODO: gpu feed_forward using gpu_matrixvector_mad
}

int nn_load_gpu(const unsigned int num_layers,
    const unsigned int *layer_offsets,
    const ActivationFunc *activation_fns,
    const double *weights,
    const unsigned int num_weights,
    const double *biases,
    const unsigned int num_biases,
    double **gpu_features,
    double **gpu_weights,
    double **gpu_layer_offsets,
    double **gpu_biases,
    ActivationFunc **gpu_activation_fns
) {
    // Assume the entire network can fit on GPU memory (for now)...
    // Copy structure, weights, and biases to the GPU.

    cudaError_t err = cudaMalloc(gpu_weights, num_weights*sizeof(double));
    if (err != cudaSuccess) {
        fprintf(stderr, "from nn_load_gpu(), cuda allocation error");
        cudaFree(*gpu_features);
        return -1;
    }

    err = cudaMalloc(gpu_layer_offsets, num_layers*sizeof(unsigned int));
    if(err != cudaSuccess) {
        fprintf(stderr, "from nn_load_gpu(), cuda allocation error");
        cudafree_ptrs(*gpu_features, *gpu_weights, NULL);
        return -1;
    }

    err = cudaMalloc(gpu_biases, num_biases*sizeof(double));
    if (err != cudaSuccess) {
        fprintf(stderr, "from nn_load_gpu(), cuda allocation error");
        cudafree_ptrs(*gpu_features, *gpu_weights, *gpu_layer_offsets, NULL);
        return -1;
    }

    err = cudaMalloc(gpu_activation_fns, num_layers*sizeof(ActivationFunc));
    if (err != cudaSuccess) {
        fprintf(stderr, "from nn_load_gpu(), cuda allocation error");
        cudafree_ptrs(*gpu_features, *gpu_weights, *gpu_layer_offsets, *gpu_biases, NULL);
        return -1;
    }

    err = cudaMemcpy(*gpu_weights, weights, num_weights*sizeof(double), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "from nn_load_gpu(), cuda memory copy error");
        cudafree_ptrs(*gpu_features, *gpu_weights, *gpu_layer_offsets, *gpu_biases, *gpu_activation_fns, NULL);
        return -1;
    }

    err = cudaMemcpy(*gpu_layer_offsets, layer_offsets, num_layers*sizeof(unsigned int), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "from nn_load_gpu(), cuda memory copy error");
        cudafree_ptrs(*gpu_features, *gpu_weights, *gpu_layer_offsets, *gpu_biases, *gpu_activation_fns, NULL);
        return -1;
    }


    err = cudaMemcpy(*gpu_biases, biases, num_biases*sizeof(double), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        fprintf(stderr, "from nn_load_gpu(), cuda memory copy error");
        cudafree_ptrs(*gpu_features, *gpu_weights, *gpu_layer_offsets, *gpu_biases, *gpu_activation_fns, NULL);
        return -1;
    }

    // Return with the gpu buffers assigned.

    return 0;
}
