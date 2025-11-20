
#include <stdio.h>
#include <stdlib.h>

#include "io_utils.h"


/*
Stores the first line from stdin in the `output` buffer. Dynamically resizes buffers 
to fit the line at runtime. Inital buffer size is determined by `default_size`.
The first line does NOT include an ending newline.

Return the size of the stored line (excluding null byte) or -1 on an error.
*/
int getline(char **output, const int default_size) {
    int c;
    int size = 0;

    // Allocate space according to the default size.
    size_t curr_alloc = default_size*sizeof(char);
    *output = malloc(curr_alloc + 1);

    // Add characters to the output buffer, resize the buffer when needed.
    while (((c = fgetc(stdin)) != '\n') && (c != EOF)) {
        // Double the size of the output buffer when needed.
        if (size*sizeof(char) + 1 > curr_alloc) {
            curr_alloc *= 2;

            char *temp = realloc(*output, curr_alloc + 1);
            if (!temp) {
                perror("from getline(), reallocation error");
                free(*output);
                return -1;
            }

            *output = temp;
        }

        (*output)[size++] = c;
    }

    (*output)[size] = '\0';

    // Allocate the correct amount of memory to store the full line.
    char *wrap = realloc(*output, size + 1);
    if (!wrap) {
        perror("from getline(), reallocation error.");
        free(*output);
        return -1;
    }

    *output = wrap;

    return size;
}