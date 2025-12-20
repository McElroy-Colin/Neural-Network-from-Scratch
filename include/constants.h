// Header file for defining constants and global variables.

#ifndef CONSTANTS_H
#define CONSTANTS_H

#include <stdio.h>

            /*        NN Constants        */

const int DEFAULT_NUM_LAYERS = 5;


            /*      Input Messages       */
            
const char *const INPUT_SUFFIX = ">  ";
const char *const INPUT_CSV_PROMPT = "Path to the input data CSV file";
const char *const INPUT_LAYERS_PROMPT = "Layer sizes";


            /*      General Constants      */

const char *const WHITESPACE = " \t\r\v\f"; // TODO: include '\n'?
const unsigned int WHITESPACE_COUNT = 5;

#endif