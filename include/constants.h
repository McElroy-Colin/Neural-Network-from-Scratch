
#include <stdio.h>

            /*        General NN Constants        */
const int DEFAULT_NUM_LAYERS = 5; // prompt hard-coded

            /*      Input Messages       */
            
const char *const INPUT_SUFFIX = ">  ";
const char *const INPUT_CSV_PROMPT = "Path to the input data CSV file";
const char *const INPUT_LAYERS_PROMPT = "Layer sizes (empty for default 5)"; // default value hard-coded