
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <stdio.h>
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

int feed_forward_krnl(const double *features,
    const unsigned int num_features,
    const unsigned int *layers,
    const unsigned int num_layers,
    const unsigned int max_layer_size,
    const ActivationFunc *activation_fns,
    const double **weights,
    const double **biases,
    double **output
) {
    // TODO: This can potentially be moved outside of the function call?
    if (!features || !layers || !activation_fns || !weights || !(*weights) || !biases || !(*biases)) {
        perror("from feed_forward_krnl(), uninitialized input pointers\n");
        return -1;
    }

    // Assume the entire network can fit on GPU memory (for now)...
    // Copy features, structure, weights, and biases to the GPU.

    double *gpu_features;
    cudaError_t err = cudaMalloc(&gpu_features, num_features*sizeof(double));
    if (err != cudaSuccess) {
        perror("from feed_forward_krnl(), cuda allocation error");
        return -1;
    }
    err = cudaMemcpy(gpu_features, features, num_features*sizeof(double), cudaMemcpyHostToDevice);
    if (err != cudaSuccess) {
        perror("from feed_forward_krnl(), cuda memory copy error");
        cudaFree(gpu_features);
        return -1;
    }

    // `layers` can be passed in each kernel call as a scalar...

    double **gpu_weights;
    err = cudaMalloc(&gpu_weights, num_layers*sizeof(double *));
    if (err != cudaSuccess) {
        perror("from feed_forward_krnl(), cuda allocation error");
        cudaFree(gpu_features);
        return -1;
    }

    double **gpu_biases;
    err = cudaMalloc(&gpu_biases, num_layers*sizeof(double *));
    if (err != cudaSuccess) {
        perror("from feed_forward_krnl(), cuda allocation error");
        cudafree_ptrs(gpu_features, gpu_weights, NULL);
        return -1;
    }

    int weight_rows = num_features;
    for (int i = 0; i < num_layers; i++) {
        // Allocate this layer's weight matrix and bias to GPU memory.
        double *curr_weights;
        err = cudaMalloc(&curr_weights, weight_rows*layers[i]*sizeof(double));
        if (err != cudaSuccess) {
            perror("from feed_forward_krnl(), cuda allocation error");
            cudaFree(gpu_features);
            cudafree_2d_darr(gpu_weights, i);
            cudafree_2d_darr(gpu_biases, i);
            return -1;
        }
        
        double *curr_biases;
        err = cudaMalloc(&curr_biases, layers[i]*sizeof(double));
        if (err != cudaSuccess) {
            perror("from feed_forward_krnl(), cuda allocation error");
            cudafree_ptrs(gpu_features, curr_weights, NULL);
            cudafree_2d_darr(gpu_weights, i);
            cudafree_2d_darr(gpu_biases, i);
            return -1;
        }

        // Copy this layer's matrix and bias to the allocated GPU memory.
        err = cudaMemcpy(curr_weights, weights[i], weight_rows*layers[i]*sizeof(double), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            perror("from feed_forward_krnl(), cuda memory copy error");
            cudafree_ptrs(gpu_features, curr_weights, curr_biases, NULL);
            cudafree_2d_darr(gpu_weights, i);
            cudafree_2d_darr(gpu_biases, i);
            return -1;
        }

        err = cudaMemcpy(curr_biases, biases[i], layers[i]*sizeof(double), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            perror("from feed_forward_krnl(), cuda memory copy error");
            cudafree_ptrs(gpu_features, curr_weights, curr_biases, NULL);
            cudafree_2d_darr(gpu_weights, i);
            cudafree_2d_darr(gpu_biases, i);
            return -1;
        }

        // Copy the flattened matrix pointer and bias to the weights array on the GPU.
        err = cudaMemcpy(gpu_weights + i, &curr_weights, sizeof(double *), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            perror("from feed_forward_krnl(), cuda memory copy error");
            cudafree_ptrs(gpu_features, curr_weights, curr_biases, NULL);
            cudafree_2d_darr(gpu_weights, i);
            cudafree_2d_darr(gpu_biases, i);
            return -1;
        }

        err = cudaMemcpy(gpu_biases + i, &curr_biases, sizeof(double *), cudaMemcpyHostToDevice);
        if (err != cudaSuccess) {
            perror("from feed_forward_krnl(), cuda memory copy error");
            cudafree_ptrs(gpu_features, curr_weights, curr_biases, NULL);
            cudafree_2d_darr(gpu_weights, i);
            cudafree_2d_darr(gpu_biases, i);
            return -1;
        }

        weight_rows = layers[i];
    }

    // TODO: Are extra buffers necessary here? space is critical to limit VRAM usage, may want to just use weights matrices directly.

    // Allocate two buffers on the GPU, each able to hold any given layer at a time.
    const unsigned int largest_layer_size = (max_layer_size > num_features) ? max_layer_size : num_features;
    double *layer_input;
    err = cudaMalloc(&layer_input, largest_layer_size*sizeof(double));
    if (err != cudaSuccess) {
        perror("from feed_forward_krnl(), cuda allocation error");
        cudaFree(gpu_features);
        cudafree_2d_darr(gpu_weights, num_layers);
        cudafree_2d_darr(gpu_biases, num_layers);
        return -1;
    }
    double *layer_output;
    err = cudaMalloc(&layer_output, largest_layer_size*sizeof(double));
    if (err != cudaSuccess) {
        perror("from feed_forward_krnl(), cuda allocation error");
        cudaFree(gpu_features);
        cudafree_2d_darr(gpu_weights, num_layers);
        cudafree_2d_darr(gpu_biases, num_layers);
        cudaFree(layer_input);
        return -1;
    }

    // TODO: Handle gpu output maybe...

    int status = 0;

    // Determine the grid layout...
    int num_blocks = 1;
    int num_threads = 1;

    compute_feed_forward<<<num_blocks, num_threads>>>(gpu_features, num_features, layers /* not necessary maybe */, num_layers, max_layer_size, activation_fns, gpu_weights, gpu_biases, output, &status);

    return status;
}

// CUDA kernel: runs on GPU
__global__ void add_vectors(const float *a, const float *b, float *c, int n) {
    int i = threadIdx.x + blockIdx.x * blockDim.x;
    if (i < n) {
        c[i] = a[i] + b[i];
    }
}

int main() {
    const int N = 16; // number of elements
    size_t size = N * sizeof(float);

    // Host arrays (CPU)
    float h_a[N], h_b[N], h_c[N];
    for (int i = 0; i < N; i++) {
        h_a[i] = i;
        h_b[i] = i*2;
    }

    // Device arrays (GPU)
    float *d_a, *d_b, *d_c;
    cudaMalloc(&d_a, size);
    cudaMalloc(&d_b, size);
    cudaMalloc(&d_c, size);

    // Copy data from host to device
    cudaMemcpy(d_a, h_a, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_b, h_b, size, cudaMemcpyHostToDevice);

    // Launch kernel with 1 block of N threads (for simplicity)
    add_vectors<<<1, N>>>(d_a, d_b, d_c, N);

    // Wait for GPU to finish
    cudaDeviceSynchronize();

    // Copy result back to host
    cudaMemcpy(h_c, d_c, size, cudaMemcpyDeviceToHost);

    // Print results
    printf("Results:\n");
    for (int i = 0; i < N; i++) {
        printf("%f + %f = %f\n", h_a[i], h_b[i], h_c[i]);
    }

    // Free GPU memory
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);

    return 0;
}
