
#include "activation_functions.cuh"

// x should be const?
__device__ double activation_func_dvc(double x, const ActivationFunc func) {
    switch (func) {
        case TEST:
            return x;
        case RELU:
            return (x > 0) ? x : 0;
        default:
            return 0.0;
    }
}