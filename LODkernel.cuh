#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>

#include "Definitions.h"
#include "RandomFunctions.cuh"
#include "BasicFunctions.cuh"
#include "Cells.cuh"
#include "Mesh.cuh"


__global__ void oxy_consumption_update(float* p_rate, Cell* cell, int num) {
	const int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= num) return;
	if (cell[i].sign == false) {
		//printf("meshupdatekernel An unexpected Dead Cell encountered:%d\n", i);
		return;
	}
	//float3 pos = cell[i].pos;
	//int3 ijk = Find_Mesh(pos, Voxel_length);
	//int meshi = Find_Index(ijk, Voxel_num);
	int meshi = cell[i].mesh_idx;
	float pO2_con = cell[i].O2_consume(); // negative
	atomicAdd(&(p_rate[meshi]), pO2_con);
}

__global__ void meshupdatekernel(float* p, float* p_rate) {
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	const int j = blockIdx.y * blockDim.y + threadIdx.y;
	const int k = blockIdx.z * blockDim.z + threadIdx.z;
	if (i >= Voxel_num || j >= Voxel_num || k >= Voxel_num) return;
	int meshi = Find_Index(i, j, k, Voxel_num);
	p[meshi] = p[meshi] / (1.f - p_rate[meshi] * Diffusion_dt);
}


// with boudary condition
__global__ void LODkernel_x(float* p, float* E, float* F) {
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int k = blockIdx.y * blockDim.y + threadIdx.y;
	if (j >= Voxel_num || k >= Voxel_num)
		return;
	float thomas_side = E[1] * F[0];
	float Y[Voxel_num] = { 0.f };
	p[Find_Index(Voxel_num - 1, j, k, Voxel_num)] = O2_Default_Concentration;
	Y[0] = O2_Default_Concentration; 

	for (int i = 1; i < Voxel_num; i++) {

		Y[i] = p[Find_Index(i, j, k, Voxel_num)] - E[i] * Y[i - 1];
	}


	for (int i = Voxel_num - 2; i > 0; i--) {
		p[Find_Index(i, j, k, Voxel_num)] = (Y[i] - thomas_side * p[Find_Index(i + 1, j, k, Voxel_num)]) / F[i];
	}
	p[Find_Index(0, j, k, Voxel_num)] = O2_Default_Concentration;
}

//__global__ void LODkernel_x_old(float* p, float* E, float* F) {
//	int j = blockIdx.x * blockDim.x + threadIdx.x;
//	int k = blockIdx.y * blockDim.y + threadIdx.y;
//	const int N = Voxel_num;
//	if (j >= N || k >= N)
//		return;
//	float thomas_side = E[1] * F[0];
//	float Y[Voxel_num] = { 0.f };
//	Y[0] = p[Find_Index(0, j, k, N)];
//	for (int i = 1; i < N - 1; i++) {
//		Y[i] = p[Find_Index(i, j, k, N)] - E[i] * Y[i - 1];
//	}
//	Y[N - 1] = p[Find_Index(N - 1, j, k, N)] - E[N - 1] * Y[N - 2];
//
//	p[Find_Index(N - 1, j, k, N)] = Y[N - 1] / F[N - 1];
//	for (int i = N - 2; i >= 0; i--) {
//		p[Find_Index(i, j, k, N)] = (Y[i] - thomas_side * p[Find_Index(i + 1, j, k, N)]) / F[i];
//	}
//}

__global__ void LODkernel_y(float* p, float* E, float* F) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int k = blockIdx.y * blockDim.y + threadIdx.y;
	if (i >= Voxel_num || k >= Voxel_num)
		return;
	float thomas_side = E[1] * F[0];
	float Y[Voxel_num] = { 0.f };
	p[Find_Index(i, Voxel_num - 1, k, Voxel_num)] = O2_Default_Concentration;
	Y[0] = O2_Default_Concentration;
	for (int j = 1; j < Voxel_num; j++) {
		Y[j] = p[Find_Index(i, j, k, Voxel_num)] - E[j] * Y[j - 1];
	}
	for (int j = Voxel_num - 2; j > 0; j--) {
		p[Find_Index(i, j, k, Voxel_num)] = (Y[j] - thomas_side * p[Find_Index(i, j + 1, k, Voxel_num)]) / F[j];
	}
	p[Find_Index(i, 0, k, Voxel_num)] = O2_Default_Concentration;
	//__syncthreads();
}

// for record old without BC
//__global__ void LODkernel_y(float* p, float* E, float* F) {
//	int i = blockIdx.x * blockDim.x + threadIdx.x;
//	int k = blockIdx.y * blockDim.y + threadIdx.y;
//	const int N = Voxel_num;
//	if (i >= N || k >= N)
//		return;
//	float thomas_side = E[1] * F[0];
//	float Y[Voxel_num] = { 0.f };
//	Y[0] = p[Find_Index(i, 0, k, N)];
//	for (int j = 1; j < N - 1; j++) {
//		// y_i = d_i - e_i * y_(i-1);
//		Y[j] = p[Find_Index(i, j, k, N)] - E[j] * Y[j - 1];
//	}
//	Y[N - 1] = p[Find_Index(i, N - 1, k, N)] - E[N - 1] * Y[N - 2];
//
//	p[Find_Index(i, N - 1, k, N)] = Y[N - 1] / F[N - 1];
//	for (int j = N - 2; j >= 0; j--) {
//		p[Find_Index(i, j, k, N)] = (Y[j] - thomas_side * p[Find_Index(i, j + 1, k, N)]) / F[j];
//	}
//	//__syncthreads();
//}

__global__ void LODkernel_z(float* p, float* E, float* F) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	if (i >= Voxel_num || j >= Voxel_num)
		return;
	float thomas_side = E[1] * F[0];
	float Y[Voxel_num] = { 0.f };

	p[Find_Index(i,j, Voxel_num - 1, Voxel_num)] = O2_Default_Concentration;
	Y[0] = O2_Default_Concentration;

	for (int k = 1; k < Voxel_num; k++) {
		// y_i = d_i - e_i * y_(i-1);
		Y[k] = p[Find_Index(i, j, k, Voxel_num)] - E[k] * Y[k - 1];
	}

	for (int k = Voxel_num - 2; k >= 0; k--) {
		p[Find_Index(i, j, k, Voxel_num)] = (Y[k] - thomas_side * p[Find_Index(i, j, k + 1, Voxel_num)]) / F[k];
	}

	p[Find_Index(i, j, 0, Voxel_num)] = O2_Default_Concentration;
	//__syncthreads();
}

__global__ void FDM_kernel(float* p,float* p_rate) {
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	const int j = blockIdx.y * blockDim.y + threadIdx.y;
	const int k = blockIdx.z * blockDim.z + threadIdx.z;
	if (i >= Voxel_num || j >= Voxel_num || k >= Voxel_num) return;
	int meshi = Find_Index(i, j, k, Voxel_num);

	float thisvalue = p[meshi];
	float coef = Diffusion_dt * O2_Diffusion_coef / Voxel_length / Voxel_length;
	float diff = p[Find_Index_clip(i - 1, j, k, Voxel_num)] + p[Find_Index_clip(i + 1, j, k, Voxel_num)] +
		p[Find_Index_clip(i, j - 1, k, Voxel_num)] + p[Find_Index_clip(i, j + 1, k, Voxel_num)] +
		p[Find_Index_clip(i, j, k - 1, Voxel_num)] + p[Find_Index_clip(i, j, k + 1, Voxel_num)] -
		6 * thisvalue;
	p[meshi] = thisvalue + coef * diff + (p_rate[meshi] - O2_Decay_rate)* thisvalue * Diffusion_dt;
	p_rate[meshi] = 0.0;
}

__global__ void Setboundary(float* p, float val) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	const int N = Voxel_num;
	if (i >= N || j >= N)
		return;
	p[Find_Index(0, i, j, N)] = val;
	p[Find_Index(N - 1, i, j, N)] = val;

	p[Find_Index(i, 0, j, N)] = val;
	p[Find_Index(i, N - 1, j, N)] = val;

	p[Find_Index(i, j, 0, N)] = val;
	p[Find_Index(i, j, N - 1, N)] = val;

	// double layer boundary
	p[Find_Index(1, i, j, N)] = val;
	p[Find_Index(N - 2, i, j, N)] = val;

	p[Find_Index(i, 1, j, N)] = val;
	p[Find_Index(i, N - 2, j, N)] = val;

	p[Find_Index(i, j, 1, N)] = val;
	p[Find_Index(i, j, N - 2, N)] = val;
}

void PreLOD(float* p_rate, Cell* cell, int num) {
	const dim3 blockSize3d(BlockWidth3d, BlockWidth3d, BlockWidth3d);
	const int gdim3d = (Voxel_num + BlockWidth3d - 1) / BlockWidth3d;
	const dim3 gridSize3d(gdim3d, gdim3d, gdim3d);
	oxy_consumption_reset << <gridSize3d, blockSize3d >> > (p_rate);
	cudaDeviceSynchronize();
	oxy_consumption_update << < (num + BlockWidth1d - 1) / BlockWidth1d, BlockWidth1d >> > (p_rate, cell, num);
	cudaDeviceSynchronize();
}

void LODsolver(float* p, float* E, float* F, float* p_rate, Cell* cell, int num) {
	const dim3 blockSize2d(BlockWidth2d, BlockWidth2d);
	const int gdim2d = (Voxel_num + BlockWidth2d - 1) / BlockWidth2d;
	const dim3 gridSize2d(gdim2d, gdim2d);

	const dim3 blockSize3d(BlockWidth3d, BlockWidth3d, BlockWidth3d);
	const int gdim3d = (Voxel_num + BlockWidth3d - 1) / BlockWidth3d;
	const dim3 gridSize3d(gdim3d, gdim3d, gdim3d);

	//oxy_consumption_update << < (num + BlockWidth1d - 1) / BlockWidth1d, BlockWidth1d >> > (p_rate, cell, num);
	//cudaDeviceSynchronize();
	Setboundary << <gridSize2d, blockSize2d >> > (p, O2_Default_Concentration);
	cudaDeviceSynchronize();

	meshupdatekernel << <gridSize3d, blockSize3d >> > (p, p_rate);
	cudaDeviceSynchronize();

	LODkernel_x << <gridSize2d, blockSize2d >> > (p, E, F);
	cudaDeviceSynchronize();
	LODkernel_y << <gridSize2d, blockSize2d >> > (p, E, F);
	cudaDeviceSynchronize();
	LODkernel_z << <gridSize2d, blockSize2d >> > (p, E, F);
	cudaDeviceSynchronize();
}


void FDMsolver(float* p, float* p_rate, Cell* cell, int num) {
	const dim3 blockSize2d(BlockWidth2d, BlockWidth2d);
	const int gdim2d = (Voxel_num + BlockWidth2d - 1) / BlockWidth2d;
	const dim3 gridSize2d(gdim2d, gdim2d);

	const dim3 blockSize3d(BlockWidth3d, BlockWidth3d, BlockWidth3d);
	const int gdim3d = (Voxel_num + BlockWidth3d - 1) / BlockWidth3d;
	const dim3 gridSize3d(gdim3d, gdim3d, gdim3d);

	//oxy_consumption_update << < (num + BlockWidth1d - 1) / BlockWidth1d, BlockWidth1d >> > (p_rate, cell, num);
	//cudaDeviceSynchronize();
	Setboundary << <gridSize2d, blockSize2d >> > (p, O2_Default_Concentration);
	cudaDeviceSynchronize();
	FDM_kernel << <gridSize3d, blockSize3d >> > (p, p_rate);
	cudaDeviceSynchronize();
}