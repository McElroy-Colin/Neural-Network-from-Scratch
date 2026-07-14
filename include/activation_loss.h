#ifndef ACTIVATION_LOSS_H
#define ACTIVATION_LOSS_H

// Different possible activations.
typedef enum {
    TEST,
    RELU
} ActivationFunc;

// Perform the given activation on the given double and return the result.
double activation_func_hst(double x, const ActivationFunc func);

// TODO: Define loss/gradient pairs as function pointers in a struct maybe?

#endif