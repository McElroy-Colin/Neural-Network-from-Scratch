#ifndef ACTIVATIONS_H
#define ACTIVATIONS_H

// Different possible activations.
typedef enum {
    TEST,
    RELU
} ActivationFunc;

// Perform the given activation on the given double and return the result.
double activation_func_hst(double x, const ActivationFunc func);

#endif