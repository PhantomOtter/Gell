#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <curand_kernel.h>
#include <iostream>
#include <stdio.h>
#include <assert.h>

#include "Definitions.h"

__host__ __device__ float bigger(float x, float y) {
	return x > y ? x : y;
}
__host__ __device__ float myabs(float x) {
	return x > 0 ? x : -x;
}

__host__ __device__ bool Check_Inside(int i,int j, int k) {
	bool inside = true;
	if (i < 0 || i >= Voxel_num || j < 0 || j >= Voxel_num || k < 0 || k >= Voxel_num) {
		inside = false;
	}
	return inside;
}

__host__ __device__ bool Check_Inside(int3 ijk) {
	return Check_Inside(ijk.x, ijk.y, ijk.z);
}

// int i,j,k ,and dimension of mesh
__host__ __device__ int Find_Index(int i, int j, int k, const int size) {
	if (i < 0 || i >= size || j < 0 || j >= size || k < 0 || k >= size) {
		printf("Find Index error:Dim %d, Input %d,%d,%d\n", size, i, j, k);
	}
	return (i + j * size + k * size * size);
}

__host__ __device__ int Find_Index(int i, int j, int k) {
	return Find_Index(i, j, k, Voxel_num);
}

__host__ __device__ int Find_Index(int3 pos, const int size) {
	if (pos.x < 0 || pos.x >= size || pos.y < 0 || pos.y >= size || pos.z < 0 || pos.z >= size) {
		printf("Find Index error:Dim %d, Input %d,%d,%d\n", size, pos.x, pos.y, pos.z);
	}
	return (pos.x + pos.y * size + pos.z * size * size);
}

__host__ __device__ int Find_Index(int3 pos) {
	return Find_Index(pos, Voxel_num);
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
//// linear transfer from point a(x,y) to b(x,y) with {y=a_y|x<a_x},{y=b_y|x>b_x}
//__host__ __device__ float Linear_step(float ax, float ay, float bx, float by, float x) {
//	if (ax <= bx)
//		return (x < ax ? ay : (x > bx ? by : (ay + (x - ax) * (by - ay) / (bx - ax))));
//	else
//		return Linear_step(bx, by, ax, ay, x);
//}

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

double CPUrand() {
	double randnumber = (double)rand() / RAND_MAX;
	return randnumber;
}

__device__ __host__ int __part1by2(int n) {
	n &= 0x000003ff;                  // base10 : 1023, binary : 1111111111, len : 10
	n = (n ^ (n << 16)) & 0xff0000ff;                  //# base10 : 4278190335, binary : 11111111000000000000000011111111, len : 32
	n = (n ^ (n << 8)) & 0x0300f00f;                  //# base10 : 50393103, binary : 11000000001111000000001111, len : 26
	n = (n ^ (n << 4)) & 0x030c30c3;                  //# base10 : 51130563, binary : 11000011000011000011000011, len : 26
	n = (n ^ (n << 2)) & 0x09249249;                  //# base10 : 153391689, binary : 1001001001001001001001001001, len : 28
	return n;
}

__device__ __host__ int __unpart1by2(int n) {
	n &= 0x09249249;                  //                 # base10 : 153391689, binary : 1001001001001001001001001001, len : 28
	n = (n ^ (n >> 2)) & 0x030c30c3;                  //# base10 : 51130563, binary : 11000011000011000011000011, len : 26
	n = (n ^ (n >> 4)) & 0x0300f00f;                  //# base10 : 50393103, binary : 11000000001111000000001111, len : 26
	n = (n ^ (n >> 8)) & 0xff0000ff;                  //# base10 : 4278190335, binary : 11111111000000000000000011111111, len : 32
	n = (n ^ (n >> 16)) & 0x000003ff;                  //# base10 : 1023, binary : 1111111111, len : 10
	return n;
}

__device__ __host__ int Morton_encode(int i, int j, int k) {
	return __part1by2(i) | (__part1by2(j) << 1) | (__part1by2(k) << 2);
	//return Find_Index(i, j, k);
}

__device__ __host__ int Morton_encode(int3 ijk) {
	return __part1by2(ijk.x) | (__part1by2(ijk.y) << 1) | (__part1by2(ijk.z) << 2);
	//return Find_Index(ijk);
}

__device__ __host__ int3 Morton_decode(int m) {
	int3 ijk = { 0,0,0 };
	ijk = { __unpart1by2(m), __unpart1by2(m >> 1), __unpart1by2(m >> 2) };
	return ijk;
}