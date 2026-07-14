#ifndef ACTIVATION_LOSS_CUH
#define ACTIVATION_LOSS_CUH

#include "activation_loss.h"

__device__ double activation_func_dvc(double x, const ActivationFunc func);

#endif