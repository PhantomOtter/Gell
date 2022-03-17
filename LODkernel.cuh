#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>

#include "Definitions.h"
#include "RandomFunctions.cuh"
#include "BasicFunctions.cuh"
#include "Cells.cuh"
#include "Mesh.cuh"


// Find_Index(x,y,z,)
__global__ void meshupdatekernel(float* p, Cell* cell, Parameters* gPara, int num) {
	const int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= num) return;
	if (cell[i].sign == false) {
		//printf("meshupdatekernel An unexpected Dead Cell encountered:%d\n", i);
		return;
	}
	float3 pos = cell[i].pos;
	int3 cellp = { 0,0,0 };
	int meshi = 0;
	float dt = gPara->Diffusion_dt;
	cellp = Find_Mesh(pos, Voxel_length);
	meshi = Find_Index(cellp.x, cellp.y, cellp.z, Voxel_num);
	float pO2_add = cell[i].O2_consume(gPara, p[meshi]) * dt;
	float currentp = atomicAdd(&(p[meshi]), pO2_add);
	//printf("Pos:%f,%f,%f,IDX:%d,%d,%d\nMeshidx:%d,Current p:%f, p add:%f\n", pos.x, pos.y, pos.z, cellp.x, cellp.y, cellp.z, meshi, currentp, pO2_add);
}

__global__ void LODkernel_x(float* p, float* E, float* F, Parameters* sp) {
	int j = blockIdx.x * blockDim.x + threadIdx.x;
	int k = blockIdx.y * blockDim.y + threadIdx.y;
	const int N = Voxel_num;
	if (j >= N || k >= N)
		return;
	float thomas_side = E[1] * F[0];
	float Y[Voxel_num] = { 0.f };
	Y[0] = p[Find_Index(0, j, k, N)];
	for (int i = 1; i < N - 1; i++) {
		// y_i = d_i - e_i * y_(i-1);
		Y[i] = p[Find_Index(i, j, k, N)] - E[i] * Y[i - 1];
	}
	Y[N - 1] = p[Find_Index(N - 1, j, k, N)] - E[N - 1] * Y[N - 2];

	p[Find_Index(N - 1, j, k, N)] = Y[N - 1] / F[N - 1];
	for (int i = N - 2; i >= 0; i--) {
		p[Find_Index(i, j, k, N)] = (Y[i] - thomas_side * p[Find_Index(i + 1, j, k, N)]) / F[i];
	}
	//__syncthreads();
}
__global__ void LODkernel_y(float* p, float* E, float* F, Parameters* sp) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int k = blockIdx.y * blockDim.y + threadIdx.y;
	const int N = Voxel_num;
	if (i >= N || k >= N)
		return;
	float thomas_side = E[1] * F[0];
	float Y[Voxel_num] = { 0.f };
	Y[0] = p[Find_Index(i, 0, k, N)];
	for (int j = 1; j < N - 1; j++) {
		// y_i = d_i - e_i * y_(i-1);
		Y[j] = p[Find_Index(i, j, k, N)] - E[j] * Y[j - 1];
	}
	Y[N - 1] = p[Find_Index(i, N - 1, k, N)] - E[N - 1] * Y[N - 2];

	p[Find_Index(i, N - 1, k, N)] = Y[N - 1] / F[N - 1];
	for (int j = N - 2; j >= 0; j--) {
		p[Find_Index(i, j, k, N)] = (Y[j] - thomas_side * p[Find_Index(i, j + 1, k, N)]) / F[j];
	}
	//__syncthreads();
}
__global__ void LODkernel_z(float* p, float* E, float* F, Parameters* sp) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	const int N = Voxel_num;
	if (i >= N || j >= N)
		return;
	float thomas_side = E[1] * F[0];
	float Y[Voxel_num] = { 0.f };
	Y[0] = p[Find_Index(i, j, 0, N)];
	for (int k = 1; k < N - 1; k++) {
		// y_i = d_i - e_i * y_(i-1);
		Y[k] = p[Find_Index(i, j, k, N)] - E[k] * Y[k - 1];
	}
	Y[N - 1] = p[Find_Index(i, j, N - 1, N)] - E[N - 1] * Y[N - 2];

	p[Find_Index(i, j, N - 1, N)] = Y[N - 1] / F[N - 1];
	for (int k = N - 2; k >= 0; k--) {
		p[Find_Index(i, j, k, N)] = (Y[k] - thomas_side * p[Find_Index(i, j, k + 1, N)]) / F[k];
	}
	//__syncthreads();
}

void LODsolver(float* p, float* E, float* F, Parameters* sp) {
	const dim3 blockSize2d(BlockWidth2d, BlockWidth2d);
	const int gdim2d = (Voxel_num + BlockWidth2d - 1) / BlockWidth2d;
	const dim3 gridSize2d(gdim2d, gdim2d);
	LODkernel_x << <gridSize2d, blockSize2d >> > (p, E, F, sp);
	cudaDeviceSynchronize();
	LODkernel_y << <gridSize2d, blockSize2d >> > (p, E, F, sp);
	cudaDeviceSynchronize();
	LODkernel_z << <gridSize2d, blockSize2d >> > (p, E, F, sp);
	cudaDeviceSynchronize();
}

__global__ void Setboundary(float* p, float val) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	const int N = Voxel_num;
	if (i >= N || j >= N)
		return;
	p[Find_Index(0, i, j, N)] = val;
	p[Find_Index(N-1, i, j, N)] = val;
	p[Find_Index(i, 0, j, N)] = val;
	p[Find_Index(i, N-1, j, N)] = val;
	p[Find_Index(i, j, 0, N)] = val;
	p[Find_Index(i, j, N-1, N)] = val;
}