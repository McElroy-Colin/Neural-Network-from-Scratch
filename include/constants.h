// Header file for defining constants and global variables.

#ifndef CONSTANTS_H
#define CONSTANTS_H

#include <stdio.h>

            /*        NN Constants        */

#define DEFAULT_NUM_LAYERS 5


            /*      device Constants       */
#define TILE_SIZE 32
    // Should be a multiple of 32 to avoid a partially-filled warp of threads.
    // Depends on shared memory space per SM, usually 48 KB at 8 bytes per double.
    // 256 is too large?


            /*      Input Messages       */
            
#define INPUT_SUFFIX ">  "
#define INPUT_CSV_PROMPT "Path to the input data CSV file"
#define INPUT_LAYERS_PROMPT "Layer sizes"


            /*      General Constants      */

#define WHITESPACE " \t\r\v\f" // TODO: include '\n'?
#define WHITESPACE_COUNT 5

#endif