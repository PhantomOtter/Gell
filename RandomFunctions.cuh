#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h" 
#include "curand_kernel.h" 
#include <iostream>
#include <time.h>
#include <cstdlib>
#include <stdlib.h>

#include "Definitions.h"


__global__ void set_random_states(curandState* curand_states)
{
    int i = threadIdx.x + blockIdx.x * blockDim.x;

    if (i < 0 || i >= Max_Cell_num)
    {
        return;
    }
    curand_init(123, i, 0, &curand_states[i]);
}
__device__ float3 randomvect(curandState* curand_states, int i, float scale = 1.f) {
    float u = curand_uniform(curand_states + i);
    float v = curand_uniform(curand_states + i);
    float theta = 2 * u * 3.1415926f;
    float phi = acosf(2.f * v - 1);
    float x = sinf(theta) * sinf(phi);
    float y = cosf(theta) * sinf(phi);
    float z = cosf(phi);
    return { x * scale,y * scale,z * scale };
}