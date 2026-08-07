// Header file for neural network training architecture.

#include "activations.h"

// Batch loss function pointer.
typedef void (*BatchLoss) (
    const double*, // True label vectors
    const double*, // Observed label vectors
    const unsigned int, // Elements per vector
    const unsigned int, // Number of vectors
    double* // Output array: will hold a loss value for each vector
);

typedef double (*PartialGrad) (double);

// TODO: Gradient function ptr.

typedef struct {
    BatchLoss loss_func;
    // grad fn here...
} ErrorGrad;

// TODO: Comment and implement
PartialGrad activation_grad_dispatch(ActivationFunc activation);

double *back_propogation( // goes only to the last layer, then we compute the final gradient in the train function
    const ActivationFunc *activation_fns,
    const double *weights,
    const unsigned int num_layers,
    const double *zs,
    const double *as,
    const unsigned int batch_size,
    double *weight_grads_out,
    double *bias_grads_out
);

// TODO: Each layer will have its own gradient that depends on that layer's activation.
//       So, train should build an array of gradient functions based on the activations...

// Compute and return the Mean-Squared Error (MSE) between two vectors `y` and `y_hat`.
// This function assumes that both vectors are length `count`.
double mean_squared_error(
    const double *y, 
    const double *y_hat, 
    const unsigned int count
);

/*
Compute the MSE for a batch of train-test vector pairs. Store the results in the `errors_out` matrix.
This function assumes that each vector in the `ys` and `y_hats` matrices is length `count_per_vec`, and 
that there are `count_per_vec*num_vecs` total elements in each.

Parameters:
    `ys`: Row-major matrix of true label vectors, where each `count_per_vec` elements is a vector with
          a total of `num_vecs` vectors. (input)
    `y_hats`: Same as `ys`, but for observed vectors. (input)
    `count_per_vec`: Length of each vector in the above matrices. (input)
    `num_vecs`: Number of vectors in the above matrices. (input)
    `errors_out`: Allocated vector of length `num_vecs` to hold each MSE value. (output)

*/
void batch_mse(
    const double *ys, 
    const double *y_hats, 
    const unsigned int count_per_vec,
    const unsigned int num_vecs,
    double *errors_out
);

/*
Optimized forward pass:
    - Instead of copying the feature vector to a buffer that is max_layer_size, just pass
      the feature vector in and do one iteration initially that puts the output in the buffer.
*/