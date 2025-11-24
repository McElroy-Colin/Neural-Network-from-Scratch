// Source file for text-related utility functions.

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#include "text_utils.h"
#include "memory_utils.h"


int char_in_str(const char* str, const unsigned int str_size, const char c) {
    for (int i = 0; i < str_size; i++) {
        if (str[i] == c) {
            return i;
        }
    }

    return -1;
}

int str_split(
    const char *str, 
    const unsigned int str_size, 
    const unsigned int default_size, 
    const char *delimeters, 
    const unsigned int delimeter_count, 
    char ***output
) {
    const char *word_start;
    size_t word_size;
    char **temp;

    unsigned int word_count = 0;

    // Allocate an array to store the substring. This can be resized as needed.
    size_t curr_alloc = default_size;
    *output = malloc(curr_alloc*sizeof(char *));
    if (!*output) {
        perror("from str_split(), allocation error");
        return -1;
    }

    int i = 0;
    while (i < str_size) {
        // Ensure that the outer output array can store all substrings by resizing it as needed.
        if (word_count >= curr_alloc) {
            curr_alloc *= 2;
            temp = realloc(*output, curr_alloc*sizeof(char *));
            if (!temp) {
                perror("from str_split(), reallocation error");
                free_2d_carr(*output, word_count);
                return -1;
            }
            *output = temp;
        }

        // Bypass inital delimeters.
        while (str[i] && (char_in_str(delimeters, delimeter_count, str[i]) > -1)) { i++; }
        // Handle when the line ends in delimeter characters.
        if (!(str[i])) { break; }

        word_start = &(str[i]);

        // Go to end of word.
        while (str[i] && (char_in_str(delimeters, delimeter_count, str[i]) == -1)) { i++; }

        word_size = (size_t)(&(str[i]) - word_start);

        // Allocate space for the word in the output inner array.
        (*output)[word_count] = malloc(word_size*sizeof(char) + 1);
        if (!((*output)[word_count])) {
            perror("from str_split(), allocation error");
            free_2d_carr(*output, word_count);
            return -1;
        }
        // Copy the word contents into the output inner array and terminate the string.
        memcpy((*output)[word_count], word_start, word_size);
        (*output)[word_count][word_size] = '\0';

        word_count++;
    }

    // Wrap the outer output array.
    if (curr_alloc > word_count) {
        temp = realloc(*output, word_count*sizeof(char *));
        if (!temp) {
            perror("from str_split(), allocation error");
            free_2d_carr(*output, word_count);
            return -1;
        }
        *output = temp;
    }

    return word_count;
}

int getline(char **output, const unsigned int default_size, FILE *stream) {
    int c;
    char *temp;
    int size = 0;

    // Allocate space according to the default size.
    size_t curr_alloc = default_size*sizeof(char);
    *output = malloc(curr_alloc + 1);

    // Add characters to the output buffer, resize the buffer when needed.
    while (((c = fgetc(stream)) != '\n') && (c != EOF)) {
        // Double the size of the output buffer when needed.
        if (size*sizeof(char) + 1 > curr_alloc) {
            curr_alloc *= 2;
            temp = realloc(*output, curr_alloc + 1);
            if (!temp) {
                perror("from getline(), reallocation error");
                free(*output);
                return -1;
            }
            *output = temp;
        }

        (*output)[size] = c;
        size++;
    }

    (*output)[size] = '\0';

    // Wrap the line in the output buffer.
    if (curr_alloc > size) {
        temp = realloc(*output, size + 1);
        if (!temp) {
            perror("from getline(), reallocation error");
            free(*output);
            return -1;
        }
        *output = temp;
    }

    return size;
}

int csv_to_matrix(
    const char *filename, 
    const unsigned int default_line_count,
    const unsigned int default_line_size,
    const unsigned int default_line_value_count,
    double ***value_output,
    unsigned int **output_lengths
) {
    FILE *file = fopen(filename, "r");
    if (!file) {
        perror("from csv_to_matrix(), could not open file");
        return -1;
    }

    char *line, *conv_end;
    int line_size, value_count;
    double **temp1;
    unsigned int *temp2;
    char **value_strs = NULL;
    unsigned int line_count = 0;

    // Allocate space for CSV rows and their lengths to be stored in the output buffers.
    // These resize dynamically when needed.
    size_t curr_alloc = default_line_count;
    *value_output = malloc(curr_alloc*sizeof(double *));
    if (!*value_output) {
        perror("from csv_to_arr(), allocation error");
        return -1;
    }
    *output_lengths = malloc(curr_alloc*sizeof(unsigned int));
    if (!*output_lengths) {
        perror("from csv_to_arr(), allocation error");
        return -1;
    }

    // Read the CSV line by line and process the text.
    // Note, this while loop ends on EOF or an empty line.
    while ((line_size = getline(&line, default_line_size, file))) {
        if (line_size == -1) {
            perror("from csv_to_matrix(), getline() error");
            free(*output_lengths);
            free_2d_darr(*value_output, line_count);
            // Only free value_strs if it was initialized in a past iteration.
            if (value_strs) {
                free_2d_carr(value_strs, value_count);
            }
            return -1;
        }
        
        // Split the line on whitespace and a comma.
        value_count = str_split(line, line_size, default_line_value_count, ", \t", 3, &value_strs);
        if (value_count == -1) {
            perror("from csv_to_matrix(), str_split() error");
            free(line);
            free(*output_lengths);
            free_2d_darr(*value_output, line_count);
            return -1;
        }

        // Store the number of values in the proper output buffer.
        (*output_lengths)[line_count] = value_count;

        // Allocate space for the CSV values to be stored in the correct inner output buffer.
        (*value_output)[line_count] = malloc(value_count*sizeof(double));
        if (!((*value_output)[line_count])) {
            perror("from csv_to_matrix(), allocation error");
            free(line);
            free(*output_lengths);
            free_2d_darr(*value_output, line_count);
            free_2d_carr(value_strs, value_count);
            return -1;
        }

        line_count++;

        // Convert each value to a double and store it in the correct inner output buffer.
        for (int i = 0; i < value_count; i++) {
            (*value_output)[line_count - 1][i] = strtod(value_strs[i], &conv_end);
            if (*conv_end != '\0') {
                perror("from csv_to_matrix(), string to double conversion error");
                free(line);
                free(*output_lengths);
                free_2d_darr(*value_output, line_count);
                free_2d_carr(value_strs, value_count);
                return -1;
            }
        }

        // Resize both output buffers if needed.
        if (line_count >= curr_alloc) {
            curr_alloc *= 2;
            temp1 = realloc(*value_output, curr_alloc*sizeof(double *));
            if (!temp1) {
                perror("from csv_to_matrix(), reallocation error");
                free(line);
                free(*output_lengths);
                free_2d_darr(*value_output, line_count);
                free_2d_carr(value_strs, value_count);
                return -1;
            }
            temp2 = realloc(*output_lengths, curr_alloc*sizeof(unsigned int));
            if (!temp2) {
                perror("from csv_to_matrix(), reallocation error");
                free(line);
                free(*output_lengths);
                free_2d_darr(*value_output, line_count);
                free_2d_carr(value_strs, value_count);
                return -1;
            }
            *value_output = temp1;
            *output_lengths = temp2;
        }

        // Free up space for the next line and set of values after each iteration.
        free(line);
        free_2d_carr(value_strs, value_count);
    }

    // Wrap the output buffers.
    if (curr_alloc > line_count) {
        temp1 = realloc(*value_output, line_count*sizeof(double *));
        if (!temp1) {
            perror("from csv_to_matrix(), allocation error");
            free(*output_lengths);
            free_2d_darr(*value_output, line_count);
            return -1;
        }
        temp2 = realloc(*output_lengths, line_count*sizeof(unsigned int));
        if (!temp2) {
            perror("from csv_to_matrix(), allocation error");
            free(*output_lengths);
            free_2d_darr(*value_output, line_count);
            return -1;
        }
        *value_output = temp1;
        *output_lengths = temp2;
    }

    return line_count;
}

int str_to_uint(const char *str, unsigned int *output) {
    unsigned int size = 0;
    unsigned int value = 0;

    // Get string length and check validity of chracters.
    while (*str) { 
        if (('0' > *str) || (*str > '9')) {
            perror("from str_to_uint(), string does must contain only digits\n");
            return -1;
        }

        str++;
        size++; 
    }

    str -= size;

    // Compute integer value assuming base-10 notation.
    for (int i = 0; i < size; i++) {
        value += (unsigned int)((str[i] - 48)*pow(10, size - (i + 1)));
    }

    *output = value;
    return size;
}