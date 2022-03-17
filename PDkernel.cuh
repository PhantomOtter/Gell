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

// followed by D.erase(thrust::remove_if(D.begin(), D.end(), is_zero()), D.end());
//__global__ void Deathkernel(Cell* cell, Parameters* gPara, curandState* curand_states, const int Num) {
//	const int i = blockDim.x * blockIdx.x + threadIdx.x;
//	if (i >= Num) return;
//	float randvalue = curand_uniform(curand_states + (i % Max_Cell_num));
//	float prob_of_death = 0.001f;
//	if (randvalue < prob_of_death) {
//		//printf("ID:%d, Death:%f \n", i, randvalue);
//		cell[i].sign = false;
//	}
//}
__global__ void Birthkernel(float* p, Cell* cell, Cell* tempcell, Parameters* gPara, curandState* curand_states, const int Num, int* newnum) {
	const int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= Num) return;
	if (cell[i].sign == false) {
		// printf("Birthkernel An unexpected Dead Cell encountered:%d\n", i);
		return;
	}
	if (*newnum >= Max_Prol_num) return;
	float randvalue = curand_uniform(curand_states + (i % Max_Cell_num));
	float dt = gPara->Biology_dt;


	float3 pos = cell[i].pos;
	int3 cellp = { 0,0,0 };
	int meshi = 0;
	cellp = Find_Mesh(pos, Voxel_length);
	meshi = Find_Index(cellp.x, cellp.y, cellp.z, Voxel_num);
	float O2 = p[meshi];

	// update cell phase
	cell[i].Volume_update(dt);
	bool proli = cell[i].Phase_update(randvalue, O2,dt);


	if (proli) {
		int idx = atomicAdd(newnum, 1);
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
		cell[i].phase = 0; //growth
		cell[i].V = cell[i].V / 2;
		cell[i].Vf = cell[i].Vf / 2;
		cell[i].Vcs = cell[i].Vcs / 2;
		cell[i].Vns = cell[i].Vns / 2;

		Cell newcell;
		newcell.sign = true;
		newcell.pos = posd;
		newcell.cell_type = cell[i].cell_type;
		newcell.phase = 0; // growth
		newcell.r = rm;
		newcell.V = cell[i].V;
		newcell.Vf = cell[i].Vf;
		newcell.Vcs = cell[i].Vcs;
		newcell.Vns = cell[i].Vns;

		tempcell[idx] = newcell;
	}
}
int CellBirth_kernel(float* p, thrust::device_vector<Cell>& GpuCell, thrust::device_vector<Cell>& TempCell, Parameters* gPara, curandState* curand_states, int currentnum, int* newnum) {
	int cnewnum = 0;
	auto GC = thrust::raw_pointer_cast(GpuCell.data());
	auto TC = thrust::raw_pointer_cast(TempCell.data());
	//std::cout << "Before Birth Current CellNum:" << currentnum << std::endl;
	Birthkernel << <(currentnum + BlockWidth1d - 1) / BlockWidth1d, BlockWidth1d >> > (p, GC, TC, gPara, curand_states, currentnum, newnum);
	cudaDeviceSynchronize();
	cudaMemcpy(&cnewnum, newnum, sizeof(int), cudaMemcpyDeviceToHost);
	//std::cout << "Birth new Cell:" << cnewnum << std::endl;
	int copyaddend = cnewnum;
	if (currentnum + cnewnum >= Max_Cell_num) {
		copyaddend = Max_Cell_num - currentnum;
	}
	try {
		thrust::copy(TempCell.begin(), TempCell.begin() + copyaddend, GpuCell.begin() + currentnum);
	}
	catch (thrust::system_error& e) {
		std::cerr << "Birth Module thrust::system_error Error: " << e.what() << std::endl;
		exit(-1);
	}
	catch (std::bad_alloc& e) {
		std::cerr << "Birth Module std::bad_alloc Error: " << e.what() << std::endl;
		exit(-1);
	}
	cudaMemset(newnum, 0, sizeof(int));
	currentnum += copyaddend;
	return currentnum;
}
int CellDeath_kernel(thrust::device_vector<Cell>& GpuCell, Parameters* gPara, curandState* curand_states, int currentnum) {
	auto GC = thrust::raw_pointer_cast(GpuCell.data());
	//Deathkernel << <(currentnum + BlockWidth1d - 1) / BlockWidth1d, BlockWidth1d >> > (GC, gPara, curand_states, currentnum);
	//cudaDeviceSynchronize();
	//std::cout << "Before Death Current CellNum:" << currentnum << std::endl;
	try {
		//GpuCell.erase(thrust::remove_if(GpuCell.begin(), GpuCell.end(), isDeadCell()), GpuCell.end());
		auto newendpointerD = thrust::remove_if(GpuCell.begin(), GpuCell.begin() + currentnum, isDeadCell());
		cudaDeviceSynchronize();
		currentnum = newendpointerD - GpuCell.begin();
		//std::cout << "After Death Current CellNum:" << currentnum << std::endl;
		//GpuCell.resize(newendpointerD - GpuCell.begin());
		//cudaDeviceSynchronize();
	}
	catch (thrust::system_error& e) {
		std::cerr << "Death Module Error: " << e.what() << std::endl;
		exit(-1);
	}
	catch (std::bad_alloc& e) {
		std::cerr << "Death Module Error: " << e.what() << std::endl;
		exit(-1);
	}
	return currentnum;
}
