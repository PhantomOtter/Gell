#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>

#include "Definitions.h"
#include "RandomFunctions.cuh"
#include "BasicFunctions.cuh"
#include "Cells.cuh"

__global__ void oxy_consumption_update(float* p_rate, Cell* cell, int num) {
	const int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= num) return;
	if (cell[i].sign == false) {
		//printf("meshupdatekernel An unexpected Dead Cell encountered:%d\n", i);
		return;
	}
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

// with Dirichlet boudary condition
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

// example without Dirichlet boundary condition directly applied
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
		Y[k] = p[Find_Index(i, j, k, Voxel_num)] - E[k] * Y[k - 1];
	}
	for (int k = Voxel_num - 2; k >= 0; k--) {
		p[Find_Index(i, j, k, Voxel_num)] = (Y[k] - thomas_side * p[Find_Index(i, j, k + 1, Voxel_num)]) / F[k];
	}
	p[Find_Index(i, j, 0, Voxel_num)] = O2_Default_Concentration;
	//__syncthreads();
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

	// double layer dirichlet boundary
	p[Find_Index(1, i, j, N)] = val;
	p[Find_Index(N - 2, i, j, N)] = val;

	p[Find_Index(i, 1, j, N)] = val;
	p[Find_Index(i, N - 2, j, N)] = val;

	p[Find_Index(i, j, 1, N)] = val;
	p[Find_Index(i, j, N - 2, N)] = val;
}

//// Dinite Difference Kernel, not used
//__global__ void FDM_kernel(float* p, float* p_rate) {
//	const int i = blockIdx.x * blockDim.x + threadIdx.x;
//	const int j = blockIdx.y * blockDim.y + threadIdx.y;
//	const int k = blockIdx.z * blockDim.z + threadIdx.z;
//	if (i >= Voxel_num || j >= Voxel_num || k >= Voxel_num) return;
//	int meshi = Find_Index(i, j, k, Voxel_num);
//
//	float thisvalue = p[meshi];
//	float coef = Diffusion_dt * O2_Diffusion_coef / Voxel_length / Voxel_length;
//	float diff = p[Find_Index_clip(i - 1, j, k, Voxel_num)] + p[Find_Index_clip(i + 1, j, k, Voxel_num)] +
//		p[Find_Index_clip(i, j - 1, k, Voxel_num)] + p[Find_Index_clip(i, j + 1, k, Voxel_num)] +
//		p[Find_Index_clip(i, j, k - 1, Voxel_num)] + p[Find_Index_clip(i, j, k + 1, Voxel_num)] -
//		6 * thisvalue;
//	p[meshi] = thisvalue + coef * diff + (p_rate[meshi] - O2_Decay_rate) * thisvalue * Diffusion_dt;
//	p_rate[meshi] = 0.0;
//}
//
//void FDMsolver(float* p, float* p_rate, Cell* cell, int num) {
//	const dim3 blockSize2d(BlockWidth2d, BlockWidth2d);
//	const int gdim2d = (Voxel_num + BlockWidth2d - 1) / BlockWidth2d;
//	const dim3 gridSize2d(gdim2d, gdim2d);
//
//	const dim3 blockSize3d(BlockWidth3d, BlockWidth3d, BlockWidth3d);
//	const int gdim3d = (Voxel_num + BlockWidth3d - 1) / BlockWidth3d;
//	const dim3 gridSize3d(gdim3d, gdim3d, gdim3d);
//
//	//oxy_consumption_update << < (num + BlockWidth1d - 1) / BlockWidth1d, BlockWidth1d >> > (p_rate, cell, num);
//	//cudaDeviceSynchronize();
//	Setboundary << <gridSize2d, blockSize2d >> > (p, O2_Default_Concentration);
//	cudaDeviceSynchronize();
//	FDM_kernel << <gridSize3d, blockSize3d >> > (p, p_rate);
//	cudaDeviceSynchronize();
//}

// middle slice where z = Voxel_num / 2
__global__ void GPU_Get_Slice(float* p, float* slice) {
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = blockIdx.y * blockDim.y + threadIdx.y;
	const int N = Voxel_num;
	if (i >= N || j >= N)
		return;
	slice[i + j * Voxel_num] = p[Find_Index(i, j, Voxel_num / 2, Voxel_num)];
}

__global__ void oxy_consumption_reset(float* p_rate) {
	const int i = blockIdx.x * blockDim.x + threadIdx.x;
	const int j = blockIdx.y * blockDim.y + threadIdx.y;
	const int k = blockIdx.z * blockDim.z + threadIdx.z;
	if (i >= Voxel_num || j >= Voxel_num || k >= Voxel_num) return;
	int meshi = Find_Index(i, j, k, Voxel_num);
	p_rate[meshi] = 0;
};

struct Mesh_struct {
	//CPU data
	float* cp = 0;
	float* cE = 0;
	float* cF = 0;
	float* cSlice = 0;
	//GPU data
	float* p = 0;
	float* p_rate = 0;
	float* E = 0;
	float* F = 0;
	float* Slice = 0;

	Mesh_struct(float concentration, float Diffusion_coef, float Decay_rate) {
		cp = (float*)malloc(Voxel_num * Voxel_num * Voxel_num * sizeof(float));
		cE = (float*)malloc(Voxel_num * sizeof(float));
		cF = (float*)malloc(Voxel_num * sizeof(float));
		cSlice = (float*)malloc(Voxel_num * Voxel_num * sizeof(float));
		printf("CPU Mesh memory is now allocated\n");

		cudaMalloc(&p, Voxel_num * Voxel_num * Voxel_num * sizeof(float));
		cudaMalloc(&p_rate, Voxel_num * Voxel_num * Voxel_num * sizeof(float));
		cudaMalloc(&E, Voxel_num * sizeof(float));
		cudaMalloc(&F, Voxel_num * sizeof(float));
		cudaMalloc(&Slice, Voxel_num * Voxel_num * sizeof(float));
		printf("GPU Mesh memory is now allocated\n");

		Mesh_initialize(concentration, Diffusion_coef, Decay_rate);

		cudaMemcpy(E, cE, Voxel_num * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(F, cF, Voxel_num * sizeof(float), cudaMemcpyHostToDevice);
		cudaMemcpy(Slice, cSlice, Voxel_num * Voxel_num * sizeof(float), cudaMemcpyHostToDevice);
		move_to_GPU();
	}
	~Mesh_struct() {
		free(cp);
		free(cE);
		free(cF);
		free(cSlice);
		cudaFree(p);
		cudaFree(p_rate);
		cudaFree(E);
		cudaFree(F);
		cudaFree(Slice);
		printf("Mesh memory is now Free\n");
	}

	void Mesh_initialize(float concentration, float Diffusion_coef, float Decay_rate) {
		for (int i = 0; i < Voxel_num * Voxel_num * Voxel_num; i++) {
			cp[i] = concentration;
		}
		for (int i = 0; i < Voxel_num * Voxel_num; i++) {
			cSlice[i] = concentration;
		}
		float thomas_side = -Diffusion_dt * Diffusion_coef / (Voxel_length * Voxel_length);
		float thomas_corner = 1.f + Diffusion_dt * Decay_rate / 3.f - thomas_side;
		float thomas_diagonal = thomas_corner - thomas_side;
		cE[0] = 0.f;
		cF[0] = thomas_corner;
		for (int i = 1; i < Voxel_num - 1; i++) {
			cE[i] = thomas_side / cF[i - 1];
			cF[i] = thomas_diagonal - cE[i] * thomas_side;
		}
		cE[Voxel_num - 1] = thomas_side / cF[Voxel_num - 2];
		cF[Voxel_num - 1] = thomas_corner - cE[Voxel_num - 1] * thomas_side;
		oxy_con_reset();
		printf("Mesh Concentration & E,F initialized\n");
	}

	void oxy_con_reset() {
		const dim3 blockSize3d(BlockWidth3d, BlockWidth3d, BlockWidth3d);
		const int gdim3d = (Voxel_num + BlockWidth3d - 1) / BlockWidth3d;
		const dim3 gridSize3d(gdim3d, gdim3d, gdim3d);
		oxy_consumption_reset << <gridSize3d, blockSize3d >> > (p_rate);
		cudaDeviceSynchronize();
	}

	// preparation before LOD kernel, get oxygen consumption map
	void PreLOD(Cell* cell, int num) {
		const dim3 blockSize3d(BlockWidth3d, BlockWidth3d, BlockWidth3d);
		const int gdim3d = (Voxel_num + BlockWidth3d - 1) / BlockWidth3d;
		const dim3 gridSize3d(gdim3d, gdim3d, gdim3d);
		oxy_consumption_reset << <gridSize3d, blockSize3d >> > (p_rate);
		cudaDeviceSynchronize();
		oxy_consumption_update << < (num + BlockWidth1d - 1) / BlockWidth1d, BlockWidth1d >> > (p_rate, cell, num);
		cudaDeviceSynchronize();
	}

	void LODsolver(Cell* cell, int num) {
		const dim3 blockSize2d(BlockWidth2d, BlockWidth2d);
		const int gdim2d = (Voxel_num + BlockWidth2d - 1) / BlockWidth2d;
		const dim3 gridSize2d(gdim2d, gdim2d);

		const dim3 blockSize3d(BlockWidth3d, BlockWidth3d, BlockWidth3d);
		const int gdim3d = (Voxel_num + BlockWidth3d - 1) / BlockWidth3d;
		const dim3 gridSize3d(gdim3d, gdim3d, gdim3d);
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

	void move_to_GPU() {
		cudaMemcpy(p, cp, Voxel_num * Voxel_num * Voxel_num * sizeof(float), cudaMemcpyHostToDevice);
		printf("CPU Mesh data moved to GPU\n");
	}
	void move_to_CPU() {
		cudaMemcpy(cp, p, Voxel_num * Voxel_num * Voxel_num * sizeof(float), cudaMemcpyDeviceToHost);
		printf("GPU Mesh data moved to CPU\n");
	}
	void Slice_to_CPU() {
		const dim3 blockSize2d(BlockWidth2d, BlockWidth2d);
		const int gdim2d = (Voxel_num + BlockWidth2d - 1) / BlockWidth2d;
		const dim3 gridSize2d(gdim2d, gdim2d);
		GPU_Get_Slice << <gridSize2d, blockSize2d >> > (p, Slice);
		cudaDeviceSynchronize();
		cudaMemcpy(cSlice, Slice, Voxel_num * Voxel_num * sizeof(float), cudaMemcpyDeviceToHost);
		printf("GPU Mesh slice data moved to CPU\n");
	}
};