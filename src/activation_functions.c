
#include "activation_functions.h"

double activation_func_hst(double x, const ActivationFunc func) {
    switch (func) {
        case TEST:
            return x;
        default:
            return 0.0;
    }
}