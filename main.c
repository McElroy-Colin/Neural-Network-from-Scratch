#include <stdio.h>
#include <stdlib.h>

#include "text_utils.h"
#include "constants.h"
#include "memory_utils.h"

#include "neural_net.h"
#include "activation_functions.h"

// Take command line or runtime user input and construct a feed-forwad neural network.
int main(int argc, char** argv) {
    char **layer_str_sizes;
    char *input_csv;
    int num_layers = 0;
    int cmd_line_csv = 0;
    int cmd_line_layers = 0;

    if (argc > 1) {
        input_csv = argv[1];
        cmd_line_csv = 1;
    // Get a CSV file path from the user.
    } else {
        printf("%s%s", INPUT_CSV_PROMPT, INPUT_SUFFIX);
        const int line_size = get_next_line(&input_csv, 20, stdin);
        if (line_size == -1) {
            fprintf(stderr, "from main(), get_next_line() error");
            return 1;
        } else if (line_size == 0) {
            fprintf(stderr, "Must enter a valid CSV path...\n");
            free(input_csv);
            return 1;
        }
    }

    if (argc > 2) {
        layer_str_sizes = &argv[2];
        cmd_line_layers = 1;
    // Get layer size values from the user if they weren't in the command line arguments.
    } else {
        char *layer_sizes_line;

        // Prompt user for layer sizes.
        printf("%s%s", INPUT_LAYERS_PROMPT, INPUT_SUFFIX);
        const int line_size = get_next_line(&layer_sizes_line, 20, stdin);
        if (line_size == -1) {
            fprintf(stderr, "from main(), get_next_line() error");
            free(input_csv);
            return 1;
        } else if (line_size == 0) {
            fprintf(stderr, "Must enter valid layer sizes...");
            free(input_csv);
            return 1;
        }

        // Split the size values string on white space to isolate numeric strings.
        num_layers = str_split(layer_sizes_line, line_size, DEFAULT_NUM_LAYERS, WHITESPACE, WHITESPACE_COUNT, &layer_str_sizes);
        if (num_layers == -1) {
            fprintf(stderr, "from main(), str_split() error");
            free(input_csv);
            return 1;
        }

        free(layer_sizes_line);
    }

    double **features;
    unsigned int *feature_lengths;
    
    // Convert the given CSV file to a matrix of doubles, so each row represents a feature vector.
    const int num_feature_vectors = csv_to_matrix(input_csv, 1, 30, 10, &features, &feature_lengths);
    if (num_feature_vectors == -1) {
        fprintf(stderr, "from main(), csv_to_matrix() error");
        return 1;
    }

    if (!cmd_line_csv) {
        free(input_csv);
    }

    // Create an array of integers to store layer size values.
    unsigned int *layer_sizes = malloc(num_layers*sizeof(unsigned int));
    if (!layer_sizes) {
        fprintf(stderr, "from main(), allocation error");
        free(input_csv);
        free_2d_carr(layer_str_sizes, num_layers);
        return 1;
    }

    // Convert string layer size values into unsigned integers.
    int conversion_error;
    for (int i = 0; i < num_layers; i++) {
        conversion_error = str_to_uint(layer_str_sizes[i], &(layer_sizes[i]));
        if (conversion_error == -1) {
            fprintf(stderr, "from main(), invalid layer size specification\n");
            free_ptrs(input_csv, layer_sizes, NULL);
            free_2d_carr(layer_str_sizes, num_layers);
            return 1;
        }
    }
    
    if (!cmd_line_layers) {
        free_2d_carr(layer_str_sizes, num_layers);
    }

    // TEST: 5 (in) -> 7 -> 4 -> 2 (out)

    const double weights[7*5 + 4*7 + 2*4] = {

        // layer input -> 1 7x5

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

    const double biases[7 + 4 + 2] = {
        2.4, 43.6, 32.34, 8.454, 45.352, 3.54, 3.561,

        2.4, 43.6, 32.34, 8.454,

        3.54, 3.561
    };

    // Assume the CSV contained vectors of the same length.
    const unsigned int num_features = feature_lengths[0];

    // Compute layer offsets before executing the network. Assume at lease 2 layers.
    unsigned int *layer_offsets[num_layers];
    layer_offsets[0] = 0;
    layer_offsets[1] = num_features*layer_sizes[1];
    for (int i = 2; i < num_layers; i++) {
        layer_offsets[i] = layer_offsets[i - 1] + layer_sizes[i - 1]*layer_sizes[i];
    }

    ActivationFunc funcs[3] = {TEST, TEST, TEST};

    double **outputs = malloc(sizeof(double *)*num_feature_vectors);
    for (int i = 0; i < num_feature_vectors; i++) {
        feed_forward_srl(
            features[i], 
            num_features, 
            layer_sizes, 
            num_layers,
            layer_offsets,
            funcs,
            weights,
            biases,
            buffer1, buffer2, // for next time...
            &outputs[i]
        );

        printf("Output f%d: (%f, %f)\n", i + 1, outputs[i][0], outputs[i][1]);
    }

    free_2d_darr(outputs, num_feature_vectors);

    free_ptrs(layer_sizes, feature_lengths, NULL);
    free_2d_darr(features, num_feature_vectors);
    return 0;
}