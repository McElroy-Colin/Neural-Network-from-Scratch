#ifndef ACTIVATION_FUNCTIONS_CUH
#define ACTIVATION_FUNCTIONS_CUH

#include "activation_functions.h"

__device__ double activation_func_dvc(double x, const ActivationFunc func);

#endif