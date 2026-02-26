#ifndef ACTIVATION_FUNCTIONS_H
#define ACTIVATION_FUNCTIONS_H

// Different possible activations.
typedef enum {
    TEST
} ActivationFunc;

// Perform the given activation on the given double and return the result.
double activation_func_srl(double x, const ActivationFunc func);

// GPU version...
#ifdef __CUDACC__
__device__
#endif
double activation_func_krnl(double x, const ActivationFunc func);

#endif