#pragma once
#include "BasicFunctions.cuh"
#include "Definitions.h"
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>

// where z=0
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
}

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
	void oxy_con_reset() {
		const dim3 blockSize3d(BlockWidth3d, BlockWidth3d, BlockWidth3d);
		const int gdim3d = (Voxel_num + BlockWidth3d - 1) / BlockWidth3d;
		const dim3 gridSize3d(gdim3d, gdim3d, gdim3d);

		oxy_consumption_reset << <gridSize3d, blockSize3d >> > (p_rate);
		cudaDeviceSynchronize();
	}
};