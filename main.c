#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <string.h>

#include "text_utils.h"
#include "constants.h"
#include "memory_utils.h"


int main(int argc, char** argv) {
    char **layer_str_sizes;
    char *input_csv;
    int num_layers = 0;
    int cmd_line_csv = 0;
    int cmd_line_layers = 0;

    if (argc > 1) {
        input_csv = argv[1];
        cmd_line_csv = 1;
    } else {
        printf("%s%s", INPUT_CSV_PROMPT, INPUT_SUFFIX);
        const int line_size = getline(&input_csv, 20, stdin);
        if (line_size == -1) {
            perror("from main(), getline() error");
            return 1;
        }
    }

    if (argc > 2) {
        layer_str_sizes = &argv[2];
        cmd_line_layers = 1;
    } else {
        char *layer_sizes_line;

        printf("%s%s", INPUT_LAYERS_PROMPT, INPUT_SUFFIX);
        const int line_size = getline(&layer_sizes_line, 20, stdin);
        if (line_size == -1) {
            perror("from main(), getline() error");
            free(input_csv);
            return 1;
        }

        num_layers = str_split(layer_sizes_line, line_size, DEFAULT_NUM_LAYERS, WHITESPACE, WHITESPACE_COUNT, &layer_str_sizes);
        if (num_layers == -1) {
            perror("from main(), str_split() error");
            free(input_csv);
            return 1;
        }

        free(layer_sizes_line);
    }

    double **features;
    unsigned int *feature_lengths;
    
    const int num_feature_vectors = csv_to_arr(input_csv, 1, 30, 10, &features, &feature_lengths);
    if (num_feature_vectors == -1) {
        perror("from main(), csv_to_arr() error");
        return 1;
    }

    for (int i = 0; i < num_feature_vectors; i++) {
        printf("Feature %d: [", i);
        for (int j = 0; j < feature_lengths[i]; j++) {
            if (j < feature_lengths[i] - 1) {
                printf("%f, ", features[i][j]);
            } else {
                printf("%f]\n", features[i][j]);
            }
        }
    }

    unsigned int *layer_sizes = malloc(num_layers*sizeof(unsigned int));
    if (!layer_sizes) {
        perror("from main(), allocation error");
        free(input_csv);
        free_2d_carr(layer_str_sizes, num_layers);
        return 1;
    }

    int conversion_error;
    
    // Convert string layer size values into unsigned integers.
    for (int i = 0; i < num_layers; i++) {
        conversion_error = str_to_uint(layer_str_sizes[i], &(layer_sizes[i]));
        if (conversion_error == -1) {
            perror("from main(), invalid layer size specification\n");
            free(input_csv);
            free_2d_carr(layer_str_sizes, num_layers);
            free(layer_sizes);
            return 1;
        }
    }
    
    if (!cmd_line_layers) {
        free_2d_carr(layer_str_sizes, num_layers);
    }

    // Actually call neural net...

    if (!cmd_line_csv) {
        free(input_csv);
    }
    free(layer_sizes);
    free_2d_darr(features, num_feature_vectors);
    return 0;
}