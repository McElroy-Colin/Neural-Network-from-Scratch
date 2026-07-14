#include <stdio.h>

#include "activation_loss.h"

// x should be const?
double activation_func_hst(double x, const ActivationFunc func) {
    switch (func) {
        case TEST:
            return x;
        case RELU:
            return (x > 0) ? x : 0;
        default:
            fprintf(stderr, "WARNING: No correct given activation function\n");
            return 0.0;
    }
}
