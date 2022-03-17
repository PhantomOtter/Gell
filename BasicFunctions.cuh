#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand_kernel.h>
#include <iostream>

#include "Definitions.h"

__host__ __device__ float bigger(float x, float y) {
	return x > y ? x : y;
}
__host__ __device__ float myabs(float x) {
	return x > 0 ? x : -x;
}
// int i,j,k ,and dimension of mesh
__host__ __device__ int Find_Index(int i, int j, int k, const int size) {
	if (i < 0 || i >= size || j < 0 || j >= size || k < 0 || k >= size) {
		printf("Find Index error:Dim %d, Input %d,%d,%d\n", size, i, j, k);
	}
	return (i + j * size + k * size * size);
}

__host__ __device__ int Find_Index(int3 pos, const int size) {
	if (pos.x < 0 || pos.x >= size || pos.y < 0 || pos.y >= size || pos.z < 0 || pos.z >= size) {
		printf("Find Index error:Dim %d, Input %d,%d,%d\n", size, pos.x, pos.y, pos.z);
	}
	return (pos.x + pos.y * size + pos.z * size * size);
}

__host__ __device__ int clip(int n, int max) {
	return n > max ? max : (n < 0 ? 0 : n);
}
__host__ __device__ int Find_Index_clip(int i, int j, int k, const int size) {
	return clip(i, size - 1) + clip(j, size - 1) * size + clip(k, size - 1) * size * size;
}

__host__ __device__ int3 Find_Mesh(float3 pos, float mesh_length) {
	int3 idx = { 0,0,0 };
	idx.x = (int)(pos.x / mesh_length);
	idx.y = (int)(pos.y / mesh_length);
	idx.z = (int)(pos.z / mesh_length);
	return idx;
}

__host__ __device__ float3 normalize(float3 dir) {
	float len = sqrtf(dir.x * dir.x + dir.y * dir.y + dir.z * dir.z);
	return { dir.x / len,dir.y / len,dir.z / len };
}
// linear transfer from point a(x,y) to b(x,y) with {y=a_y|x<a_x},{y=b_y|x>b_x}
__host__ __device__ float Linear_step(float ax, float ay, float bx, float by, float x) {
	if (ax <= bx)
		return (x < ax ? ay : (x > bx ? by : (ay + (x - ax) * (by - ay) / (bx - ax))));
	else
		return Linear_step(bx, by, ax, ay, x);
}

void showGPUinfo(bool showall = false) {
	if (showall) {
		size_t limit = 0;
		cudaDeviceGetLimit(&limit, cudaLimitStackSize);
		printf("cudaLimitStackSize: %u KB\n", (unsigned)limit / 1024);
		cudaDeviceGetLimit(&limit, cudaLimitPrintfFifoSize);
		printf("cudaLimitPrintfFifoSize: %u KB\n", (unsigned)limit / 1024);
		cudaDeviceGetLimit(&limit, cudaLimitMallocHeapSize);
		printf("cudaLimitMallocHeapSize: %u MB \n", (unsigned)limit / 1024 / 1024);
	}
	size_t total_gpu_bytes;
	size_t free_gpu_bytes;
	cudaMemGetInfo(&free_gpu_bytes, &total_gpu_bytes);
	std::cout << "Free Gpu Mem: " << free_gpu_bytes / 1024 / 1024 << " MB" << std::endl;
	std::cout << "Total Gpu Mem: " << total_gpu_bytes / 1024 / 1024 << " MB" << std::endl;
}


struct isMinusOne
{
	__host__ __device__
		bool operator()(const int x)
	{
		return (x == -1);
	}
};