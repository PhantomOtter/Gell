#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h" 
#include "curand_kernel.h" 
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/remove.h>
#include <iostream>
#include <time.h>

#include "BasicFunctions.cuh"
#include "RandomFunctions.cuh"
#include "Definitions.h"
#include "Cells.cuh"

// remember to adjust savecsv function to record the result
//__global__ void findcloest(Cell* cell, Parameters* gPara, int num) {
//	const int i = blockDim.x * blockIdx.x + threadIdx.x;
//	if (i >= num) return;
//	if (cell[i].sign == false) {
//		printf("findcloest An unexpected Dead Cell encountered:%d\n", i);
//		return;
//	}
//	float3 pos1 = cell[i].pos;
//	float3 pos2 = { 0.f,0.f,0.f };
//	float dtemp = 0.f;
//	float thres = 100.f;
//	for (int j = 0; j < num; j++) {
//		if (j == i || cell[j].sign == false) continue;
//		pos2 = cell[j].pos;
//		dtemp = pos1.x - pos2.x;
//		dtemp = (dtemp > 0) ? dtemp : -dtemp;
//		if (dtemp >= thres) continue;
//		dtemp = pos1.y - pos2.y;
//		dtemp = (dtemp > 0) ? dtemp : -dtemp;
//		if (dtemp >= thres) continue;
//		dtemp = pos1.z - pos2.z;
//		dtemp = (dtemp > 0) ? dtemp : -dtemp;
//		if (dtemp >= thres) continue;
//		dtemp = sqrtf((pos1.x - pos2.x) * (pos1.x - pos2.x) +
//			(pos1.y - pos2.y) * (pos1.y - pos2.y) + (pos1.z - pos2.z) * (pos1.z - pos2.z));
//		if (dtemp < thres) {
//			thres = dtemp;
//		}
//	}
//	cell[i].closest = thres;
//}