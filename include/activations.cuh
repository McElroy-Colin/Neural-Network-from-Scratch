#ifndef ACTIVATIONS_CUH
#define ACTIVATIONS_CUH

#include "activations.h"

__device__ double activation_func_dvc(double x, const ActivationFunc func);

#endif