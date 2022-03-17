#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h" 
#include "curand_kernel.h" 
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/remove.h>
#include <thrust/copy.h>
#include <thrust/sort.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <iostream>
#include <time.h>

#include "BasicFunctions.cuh"
#include "RandomFunctions.cuh"
#include "Definitions.h"
#include "Cells.cuh"


__global__ void updateCellMeshIdx(Cell* cell, int num) {
	const int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= num) return;
	if (cell[i].sign == false) {
		// printf("updateCellMeshIdx An unexpected Dead Cell encountered:%d\n", i);
		return;
	}
	int3 ijk = Find_Mesh(cell[i].pos, Voxel_length);
	if (ijk.x >= Voxel_num || ijk.y >= Voxel_num || ijk.z >= Voxel_num) {
		cell[i].sign = false;
		return;
	}
	int idx = Find_Index(ijk, Voxel_num);
	cell[i].posmeshidx = idx;
}

//if cell[i].cellmeshidx - cell[i - 1].cellmeshidx > 0, then i is a start index this cell mesh, record the cellidx of i in the array, otherwise make it -1
// go over cell num
__global__ void Generate_CellIndex_of_StartCell(Cell* cell, int* array, int num) {
	const int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= num) return;
	if (i == 0) {
		array[i] = 0;
		return;
	}
	int check = cell[i].posmeshidx - cell[i - 1].posmeshidx;
	if (check < 0) printf("Error in the Find_CellIndex_of_StartCell, cell sort error, %d\n", i);
	if (check == 0) {
		array[i] = -1;
	}
	if (check > 0) {
		array[i] = i;
	}
}
// go over len of start cell array (short)
// pushback current cell num
__global__ void Generate_CellIndex_of_StartCell_of_CellMesh(Cell* cell, int* CellIndex_of_StartCell_of_each_CellMesh, int* CellIndex_of_StartCell, int numofstartcells, int cellnum) {
	const int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= numofstartcells) return;
	int thiscellidx = CellIndex_of_StartCell[i];  //first element is always 0
	int thismeshidx = cell[thiscellidx].posmeshidx;
	int startmeshidx = -1;
	if (i == 0) {
		for (int meshidx = startmeshidx + 1; meshidx <= thismeshidx; meshidx++) {
			CellIndex_of_StartCell_of_each_CellMesh[meshidx] = 0;
		}
	}
	if (i > 0) {
		startmeshidx = cell[CellIndex_of_StartCell[i - 1]].posmeshidx;
		for (int meshidx = startmeshidx + 1; meshidx <= thismeshidx; meshidx++) {
			CellIndex_of_StartCell_of_each_CellMesh[meshidx] = thiscellidx;
		}
		if (i == (numofstartcells - 1)) {
			for (int exmeshidx = thismeshidx + 1; exmeshidx <= Voxel_num * Voxel_num * Voxel_num; exmeshidx++) {
				CellIndex_of_StartCell_of_each_CellMesh[exmeshidx] = cellnum;
			}
		}
	}

}

// example of loop over mesh
__device__ void loopofmeshi(int* CellIndex_of_StartCell_of_CellMesh, int cellmeshidx) {
	int cellstart = CellIndex_of_StartCell_of_CellMesh[cellmeshidx];
	int cellend = CellIndex_of_StartCell_of_CellMesh[cellmeshidx + 1];
	for (int i = cellstart; i < cellend; i++) {
		printf("Cell in Meshidx %d: %d\n", cellmeshidx, i);
	}
}

// The final V3 force kernel
__global__ void forcekernel(Cell* cell, Parameters* gPara, int* CellIndex_of_StartCell_of_CellMesh, int num) {
	const int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= num) return;
	if (cell[i].sign == false) {
		// printf("V3_forcekernel An unexpected Dead Cell encountered:%d\n", i);
		return;
	}
	float3 pos1 = cell[i].pos;
	float r1 = cell[i].r;
	float3 pos2 = { 0.f,0.f,0.f };
	float r2 = 0.f;
	int3 posidx = Find_Mesh(pos1, Voxel_length);

	float distance = 0.f;
	float fx = 0, fy = 0, fz = 0;
	float R = 0.f;
	float Ra_ratio = gPara->Ra_ratio, Ccca = gPara->Ccca, Cccr = gPara->Cccr;
	float thres = gPara->Tumor_CC_Max_Dist;
	float coefr = 0.f;
	float coefa = 0.f;

	for (int xi = posidx.x - 1; xi <= posidx.x + 1; xi++) {
		if (xi < 0 || xi >= Voxel_num) continue;
		for (int yi = posidx.y - 1; yi <= posidx.y + 1; yi++) {
			if (yi < 0 || yi >= Voxel_num) continue;
			for (int zi = posidx.z - 1; zi <= posidx.z + 1; zi++) {
				if (zi < 0 || zi >= Voxel_num) continue;
				int meshidx = Find_Index(xi, yi, zi, Voxel_num);
				int cellstart = CellIndex_of_StartCell_of_CellMesh[meshidx];
				int cellend = CellIndex_of_StartCell_of_CellMesh[meshidx + 1];
				for (int cell2idx = cellstart; cell2idx < cellend; cell2idx++) {
					if (cell[cell2idx].sign == false) continue;
					pos2 = cell[cell2idx].pos;

					if (myabs(pos1.x - pos2.x) >= thres) continue;
					if (myabs(pos1.y - pos2.y) >= thres) continue;
					if (myabs(pos1.z - pos2.z) >= thres) continue;
					r2 = cell[cell2idx].r;
					// now thres = Ra1+Ra2
					R = bigger(r2 + r1, 0.1f);
					thres = R * Ra_ratio;
					distance = sqrtf((pos1.x - pos2.x) * (pos1.x - pos2.x) +
						(pos1.y - pos2.y) * (pos1.y - pos2.y) + (pos1.z - pos2.z) * (pos1.z - pos2.z) + 0.01);
					if (distance >= thres) continue;
					//force from j(pos2) to i(pos1)
					//repulsive
					
					coefr = (distance <= R) ? Cccr * ((1 - distance / R) * (1 - distance / R) / distance) : 0.f;
					coefa = (distance <= thres) ? Ccca * ((1 - distance / thres) * (1 - distance / thres) / distance) : 0.f;
					fx += (pos1.x - pos2.x) * (coefr - coefa);
					fy += (pos1.y - pos2.y) * (coefr - coefa);
					fz += (pos1.z - pos2.z) * (coefr - coefa);
				}
			}
		}

	}
	cell[i].force = { fx,fy,fz };
}

__global__ void movementkernel(Cell* cell, Parameters* gPara, int num) {
	const int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= num) return;
	if (cell[i].sign == false) {
		//printf("movementkernel An unexpected Dead Cell encountered:%d\n", i);
		return;
	}
	float dt = gPara->Biology_dt;
	float3 pos = cell[i].pos;
	float3 of = cell[i].oldforce;
	float3 f = cell[i].force;
	float3 move = { 0.f,0.f,0.f };
	move.x = dt / 2.f * (3.f * f.x - of.x);
	move.y = dt / 2.f * (3.f * f.y - of.y);
	move.z = dt / 2.f * (3.f * f.z - of.z);
	pos.x += move.x;
	pos.y += move.y;
	pos.z += move.z;

	//printf("Cell %d, F(%f£¬%f£¬%f),OF(%f£¬%f£¬%f),Move(%f£¬%f£¬%f)\n", i, f.x, f.y, f.z, of.x, of.y, of.z, move.x, move.y, move.z);
	//ÒÆ³ýÔ½½çÏ¸°û
	float volumelimit = Volume_length;
	if (pos.x<0.f || pos.x>=volumelimit || pos.y<0.f || pos.y>=volumelimit ||
		pos.z<0.f || pos.z>=volumelimit) {
		cell[i].sign = false;
		return;
	}
	cell[i].pos.x = pos.x;
	cell[i].pos.y = pos.y;
	cell[i].pos.z = pos.z;
	cell[i].oldforce = f;
	cell[i].force = { 0.f,0.f,0.f };
}

void Cellmesh_FM_kernel(thrust::device_vector<Cell>& GpuCell, thrust::device_vector<int>& CellIndex_of_StartCell_of_CellMesh, Parameters* gPara,int currentnum) {
	//Force and Movement Module
	//currentnum = GpuCell.size();

	auto GC = thrust::raw_pointer_cast(GpuCell.data());
	updateCellMeshIdx << <(currentnum + BlockWidth1d - 1) / BlockWidth1d, BlockWidth1d >> > (GC, currentnum);
	cudaDeviceSynchronize();

	try {
		thrust::sort(GpuCell.begin(), GpuCell.begin() + currentnum);
		cudaDeviceSynchronize();
	}
	catch (thrust::system_error& e) {
		std::cerr << "FM Module Sort thrust::system_error Error: " << e.what() << std::endl;
		exit(-1);
	}
	catch (std::bad_alloc& e) {
		std::cerr << "FM Module Sort std::bad_alloc Error: " << e.what() << std::endl;
		exit(-1);
	}

	GC = thrust::raw_pointer_cast(GpuCell.data());
	//have the length of cell num, -1 if is not a start cell of a mesh
	thrust::device_vector<int> CellIndex_of_StartCell(currentnum);
	//CellIndex_of_StartCell.resize(currentnum);
	cudaDeviceSynchronize();

	auto pointerA = thrust::raw_pointer_cast(CellIndex_of_StartCell.data());
	Generate_CellIndex_of_StartCell << <(currentnum + BlockWidth1d - 1) / BlockWidth1d, BlockWidth1d >> > (GC, pointerA, currentnum);
	cudaDeviceSynchronize();


	try {
		CellIndex_of_StartCell.erase(thrust::remove(CellIndex_of_StartCell.begin(), CellIndex_of_StartCell.end(), -1), CellIndex_of_StartCell.end());
		cudaDeviceSynchronize();
	}
	catch (thrust::system_error& e) {
		std::cerr << "FM Module remove/erase CellIndex_of_StartCell erase Error: " << e.what() << std::endl;
		exit(-1);
	}
	catch (std::bad_alloc& e) {
		std::cerr << "FM Module remove/erase CellIndex_of_StartCell erase Error: " << e.what() << std::endl;
		exit(-1);
	}


	//the length of this array is changed
	int Startcell_arraylen = CellIndex_of_StartCell.size();
	// from 0 to Startcell_arraylen, last thread make all the element afterwards currentnum
	auto pointerB = thrust::raw_pointer_cast(CellIndex_of_StartCell_of_CellMesh.data()); // len Voxel_num* Voxel_num* Voxel_num+1
	pointerA = thrust::raw_pointer_cast(CellIndex_of_StartCell.data());  //len Startcell_arraylen
	GC = thrust::raw_pointer_cast(GpuCell.data());
	cudaDeviceSynchronize();

	//===================================
	Generate_CellIndex_of_StartCell_of_CellMesh << <(Startcell_arraylen + BlockWidth1d - 1) / BlockWidth1d, BlockWidth1d >> > (GC, pointerB, pointerA, Startcell_arraylen, currentnum);
	//===================================
	cudaDeviceSynchronize();


	GC = thrust::raw_pointer_cast(GpuCell.data());
	pointerB = thrust::raw_pointer_cast(CellIndex_of_StartCell_of_CellMesh.data());
	cudaDeviceSynchronize();

	//SM_forcekernel << <(currentnum + BlockWidth1d - 1) / BlockWidth1d, BlockWidth1d >> > (GC, gcp, gsp, currentnum);
	forcekernel << <(currentnum + BlockWidth1d - 1) / BlockWidth1d, BlockWidth1d >> > (GC, gPara, pointerB, currentnum);
	cudaDeviceSynchronize();
	movementkernel << <(currentnum + BlockWidth1d - 1) / BlockWidth1d, BlockWidth1d >> > (GC, gPara, currentnum);
	cudaDeviceSynchronize();
}