#include "train.h"

double mean_squared_error(const double *y, const double *y_hat, const unsigned int count) {
        double sum = 0.0;
        for (int i = 0; i < count; i++) {
            double diff = y_hat[i] - y[i];
            sum += diff*diff;
        }

        // Additional factor of `1/2` to make the gradient cleaner for training.
        return sum / (2*count);
    }

void batch_mse(const double *ys,
    const double *y_hats, 
    const unsigned int count_per_vec,
    const unsigned int num_vecs,
    double *errors_out) {

    double *curr_ys = ys;
    double *curr_y_hats = y_hats;

    for (int i = 0; i < num_vecs; i++) {
        errors_out[i] = mean_squared_error(curr_ys, curr_y_hats, count_per_vec);
        curr_ys += count_per_vec;
        curr_y_hats += count_per_vec;
    }

    return;
}

int train_hst( // TODO: make what error function to use an argument of the function call, use function pointers
    NeuralNetwork *neural_net,
    const double *feature_vectors, // Flattened array of ALL feature vectors (length num_features*num_vectors)
    const double *ys, // flattened array of output vectors where the i'th corresponds to the i'th feature vector^, length output_size*num_vectors
    const unsigned int num_vectors, // number of training vectors used
    const unsigned int batch_size, // number of feature vectors per batch
    const ErrorGrad *loss_gradient
) {

    if (batch_size == 0) {
        fprintf(stderr, "from train_hst(), batch_size must be > 0\n");
        return -1;
    } else if (num_vectors == 0) {
        fprintf(stderr, "from train_hst(), num_vectors must be > 0\n");
        return -1;
    }

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
    
    // Ceiling division to get the number of batches.
    const unsigned int num_batches = (num_vectors + batch_size - 1) / batch_size;
    // Length of the neural network's output layer.
    const unsigned int output_size = layers[num_layers - 1];
    
    // Buffer to store outputs of a batch's forward pass.
    /*
    Note, `y_hats` is used as both y_hats and the flattened array of forward pass outputs for the batch.
    So, `y_hats` must always have enough room for `max_layer_size`, even on the last output of 
    the batch. So, add `max_layer_size` elements and subtract a `output_size`, so the final `max_layer_size` 
    elements will fit the final output vector AND each layer on the final forward pass.

    `batch_size*output_size + max_layer_size - output_size => (batch_size - 1)*output_size + max_layer_size`
    */
    double *y_hats = malloc(((batch_size - 1)*output_size + max_layer_size)*sizeof(double));
    if (!y_hats) {
        fprintf(stderr, "from train_hst(), memory allocation error\n");
        return -1;
    }
    double *buffer2 = malloc(max_layer_size*sizeof(double));
    if (!buffer2) {
        fprintf(stderr, "from train_hst(), memory allocation error\n");
        free(y_hats);
        return -1;
    }
    double *errors_out = malloc(batch_size*sizeof(double));
    if (!buffer2) {
        fprintf(stderr, "from train_hst(), memory allocation error\n");
        free_ptrs(y_hats, buffer2, NULL);
        return -1;
    }

    // Outer loop sends batches of feature vectors.
    for (int b = 0; b < num_batches; b++) {
        double *curr_yhats = y_hats;
        // Forward pass loop for the current batch.
        for (int fp = 0; fp < batch_size; fp++) { // TODO: last batch is usually smaller, so last iteration could be faster.
            // Give the `curr_yhats` buffer the current feature vector.
            memcpy(curr_yhats, feature_vectors, num_features*sizeof(double));

            // `curr_yhats` will now hold the current forward pass output. 
            // TODO: Write a faster training version of the forward pass, caching intermediate values used in backprop.
            feed_forward_hst(neural_net, curr_yhats, buffer2);
            
            // `feed_forward_hst` populates the next `output_size` elements of `curr_yhats`.
            curr_yhats += output_size;
            // Increment to the next feature vector.
            feature_vectors += num_features;
        }
        // Get error values for each output vector of the batch.
        batch_mse(ys, y_hats, output_size, batch_size, errors_out); // TODO choice argument...

        // NEXT: compute gradient and adjust weights.
        // ...

        y_hats = curr_yhats;
    }


    free_ptrs(y_hats, buffer2, errors_out, NULL);
}