
#include "activation_functions.h"

__device__ double activation_func_krnl(double x, const ActivationFunc func) {
    switch (func) {
        case TEST:
            return x;
        default:
            return 0.0;
    }
}