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

// kernel for proliferation and death

struct isDeadCell
{
	__host__ __device__
		bool operator()(const Cell x)
	{
		return (x.sign == false);
	}
};

__global__ void Birthkernel(float* p, Cell* cell, curandState* curand_states, int currentnum, int* gcurrentnum) {
	const int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= currentnum) return;
	if (cell[i].sign == false) return;
	if (*gcurrentnum >= Max_Cell_num) return;
	float randvalue = curand_uniform(curand_states + (i % Max_Cell_num));

	float3 pos = cell[i].pos;
	int3 cellp = { 0,0,0 };
	int meshi = 0;
	cellp = Find_Mesh(pos, Voxel_length);
	meshi = Find_Index(cellp.x, cellp.y, cellp.z, Voxel_num);
	float O2 = p[meshi];

	// update cell phase
	cell[i].Volume_update();
	bool proli = cell[i].Phase_update_ki67adv(randvalue, O2);   // modified to agree with physicell
	////0 for mature (quiescent)
	////1 divide reparation (premitotic)
	////2 for growing (postmitotic)
	////3 for apoptotic
	////4 for early necrotic
	////5 for late necrotic

	if (proli && (*gcurrentnum < Max_Cell_num-1)) {
		int idx = atomicAdd(gcurrentnum, 1);

		if (idx < Max_Cell_num) {
			//printf("ID:%d, Idx:%d, Birth:%f \n", i, idx, randvalue);
			float3 dir = randomvect(curand_states, (i % Max_Cell_num));
			float3 posm = cell[i].pos;
			float rm = cell[i].r;
			float distance = 0.2063f * rm;
			rm *= 0.7937f;
			float3 posd = { posm.x + dir.x * distance,posm.y + dir.y * distance, posm.z + dir.z * distance };
			posm = { posm.x - dir.x * distance,posm.y - dir.y * distance, posm.z - dir.z * distance };

			cell[i].r = rm;
			cell[i].pos = posm;
			cell[i].phase = 2; // postmitotic
			cell[i].cell_clock = 0;
			cell[i].V = cell[i].V / 2;
			cell[i].Vf = cell[i].Vf / 2;
			cell[i].Vcs = cell[i].Vcs / 2;
			cell[i].Vns = cell[i].Vns / 2;

			Cell newcell;
			newcell.sign = true;
			newcell.pos = posd;
			newcell.phase = 2; // postmitotic
			newcell.cell_clock = 0;
			newcell.r = rm;
			newcell.V = cell[i].V;
			newcell.Vf = cell[i].Vf;
			newcell.Vcs = cell[i].Vcs;
			newcell.Vns = cell[i].Vns;

			cell[idx] = newcell;
		}
	}
}
int CellBirth_kernel(float* p, thrust::device_vector<Cell>& GpuCell, curandState* curand_states, int currentnum, int* gcurrentnum) {
	int newnum = 0;
	auto GC = thrust::raw_pointer_cast(GpuCell.data());
	//std::cout << "Before Birth Current CellNum:" << currentnum << std::endl;
	Birthkernel << <(currentnum + BlockWidth1d - 1) / BlockWidth1d, BlockWidth1d >> > (p, GC,curand_states, currentnum, gcurrentnum);
	cudaDeviceSynchronize();

	cudaMemcpy(&newnum, gcurrentnum, sizeof(int), cudaMemcpyDeviceToHost);
	//std::cout << "Birth_kernel Current Cell:" << currentnum << ", New num:" << newnum <<std::endl;
	return newnum;
}

int CellDeath_kernel(thrust::device_vector<Cell>& GpuCell, curandState* curand_states, int currentnum, int* gcurrentnum) {
	auto GC = thrust::raw_pointer_cast(GpuCell.data());
	try {
		auto newendpointerD = thrust::remove_if(GpuCell.begin(), GpuCell.begin() + currentnum, isDeadCell());
		cudaDeviceSynchronize();
		currentnum = newendpointerD - GpuCell.begin();
	}
	catch (thrust::system_error& e) {
		std::cerr << "Death Module Error: " << e.what() << std::endl;
		exit(-1);
	}
	catch (std::bad_alloc& e) {
		std::cerr << "Death Module Error: " << e.what() << std::endl;
		exit(-1);
	}
	cudaMemcpy(gcurrentnum, &currentnum, sizeof(int), cudaMemcpyHostToDevice);
	return currentnum;
}
