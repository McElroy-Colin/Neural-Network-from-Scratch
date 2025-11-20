
#include <stdio.h>
#include <stdlib.h>
#include <ctype.h>
#include <string.h>

#include "io_utils.h"
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
        const int line_size = getline(&input_csv, 20);
        if (line_size == -1) {
            perror("from main(), getline() allocation error");
            return 1;
        }
    }

    if (argc > 2) {
        layer_str_sizes = &argv[2];
        cmd_line_layers = 1;
    } else {
        char *layer_sizes_line;

        printf("%s%s", INPUT_LAYERS_PROMPT, INPUT_SUFFIX);
        const int line_size = getline(&layer_sizes_line, 20);
        if (line_size == -1) {
            perror("from main(), getline() allocation error");
            free(input_csv);
            return 1;
        }

        char *word_start;
        size_t word_size;
        size_t curr_alloc = DEFAULT_NUM_LAYERS;
        layer_str_sizes = malloc(curr_alloc*sizeof(char *));

        int i = 0;
        while (i < line_size) {
            if (num_layers > curr_alloc) {
                curr_alloc *= 2;
                char **temp = realloc(layer_str_sizes, curr_alloc*sizeof(char *));
                if (!temp) {
                    perror("from main(), reallocation error");
                    free(input_csv);
                    free(layer_sizes_line);
                    free_2d_arr((void **)layer_str_sizes, num_layers);
                    return 1;
                }

                layer_str_sizes = temp;
            }

            // Bypass inital whitespace.
            while (layer_sizes_line[i] && isspace((unsigned char)(layer_sizes_line[i]))) {
                i++; 
            }
            if (!(layer_sizes_line[i])) {
                break; 
            }
            word_start = &(layer_sizes_line[i]);
            // Go to end of word.
            while (layer_sizes_line[i] && !isspace((unsigned char)(layer_sizes_line[i]))) {
                i++; 
            }

            word_size = (size_t)(&(layer_sizes_line[i]) - word_start);

            layer_str_sizes[num_layers] = malloc(word_size*sizeof(char) + 1);
            memcpy(layer_str_sizes[num_layers], word_start, word_size);
            layer_str_sizes[num_layers++][word_size] = '\0';
        }

        free(layer_sizes_line);

        char **temp = realloc(layer_str_sizes, num_layers*sizeof(char *));
        if (!temp) {
            perror("from main(), reallocation error");
            free(input_csv);
            free_2d_arr((void **)layer_str_sizes, num_layers);
            return 1;
        }

        layer_str_sizes = temp;
    }

    for (int i = 0; i < num_layers; i++) {
        printf("|%s|", layer_str_sizes[i]);
    }

    // convert layer_str_sizes to int list

    if (!cmd_line_layers) {
        free_2d_arr((void **)layer_str_sizes, num_layers);
    }
    if (!cmd_line_csv) {
        free(input_csv);
    }

    return 0;
}