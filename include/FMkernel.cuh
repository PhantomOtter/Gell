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
#include <thrust/execution_policy.h>
#include <iostream>
#include <time.h>

#include "BasicFunctions.cuh"
#include "RandomFunctions.cuh"
#include "Definitions.h"
#include "Cells.cuh"


__global__ void updateCellMeshIdx(Cell* cell, int cnum, int* key) {
	const int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= cnum) return;
	int3 ijk = Find_Mesh(cell[i].pos, Voxel_length);
	if (!Check_Inside(ijk)) {
		cell[i].sign = false;
		key[i] = Sort_Key_Last;
		//cell[i].morton_code = Morton_encode(Voxel_num, Voxel_num, Voxel_num) + 1;
		//cell[i].mesh_idx = Voxel_num * Voxel_num * Voxel_num;
		return;
	}
	key[i] = Morton_encode(ijk);//Find_Index(ijk, Voxel_num); // Morton_encode(ijk);
	// cell[i].morton_code = Morton_encode(ijk);
	cell[i].mesh_idx = Find_Index(ijk, Voxel_num); // Find_Index(ijk, Voxel_num);
}

__global__ void Mech_Mesh_se_Reset(int* s, int* e, int num) {
	const int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= num) return;
	s[i] = Max_Cell_num + 1;
	e[i] = -1;
}

__global__ void Mech_Mesh_se_Get(Cell* cell, int cnum, int* s, int* e) {
	const int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= cnum || cell[i].sign == false) return;
	int mesh_idx = cell[i].mesh_idx;
	if (mesh_idx < 0 || mesh_idx >= Voxel_num * Voxel_num * Voxel_num) return;

	atomicMax(&e[mesh_idx], i);
	atomicMin(&s[mesh_idx], i);
}

// The final V3 force kernel
__global__ void forcekernel(Cell* cell, int num, int* s, int* e) {
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
	float thres = Tumor_CC_Max_Dist;
	float coefr = 0.f;
	float coefa = 0.f;

	for (int xi = posidx.x - 1; xi <= posidx.x + 1; xi++) {
		for (int yi = posidx.y - 1; yi <= posidx.y + 1; yi++) {
			for (int zi = posidx.z - 1; zi <= posidx.z + 1; zi++) {
				if (!Check_Inside(xi, yi, zi)) continue;

				int meshidx = Find_Index(xi, yi, zi);
				int cellstart = s[meshidx];
				int cellend = e[meshidx];
				if (cellend < 0) continue;
				for (int cell2idx = cellstart; cell2idx <= cellend; cell2idx++) {
					if (cell[cell2idx].sign == false) continue;

					pos2 = cell[cell2idx].pos;
					distance = sqrtf((pos1.x - pos2.x) * (pos1.x - pos2.x) +
						(pos1.y - pos2.y) * (pos1.y - pos2.y) + (pos1.z - pos2.z) * (pos1.z - pos2.z) + 0.0001);
					if (distance >= thres ) continue;

					///////////////////////////////////////////////////
					// costomize your cell-cell interaction here

					r2 = cell[cell2idx].r;
					// now thres = Ra1+Ra2
					R = bigger(r2 + r1, 0.01f);
					thres = R * Ra_ratio;

					if (distance >= thres) continue;
					//force from j(pos2) to i(pos1)
					//repulsive

					coefr = (distance <= R) ? Cccr * ((1 - distance / R) * (1 - distance / R) / distance) : 0.f;
					coefa = (distance <= thres) ? Ccca * ((1 - distance / thres) * (1 - distance / thres) / distance) : 0.f;
					fx += (pos1.x - pos2.x) * (coefr - coefa);
					fy += (pos1.y - pos2.y) * (coefr - coefa);
					fz += (pos1.z - pos2.z) * (coefr - coefa);

					///////////////////////////////////////////////////
				}
			}
		}

	}
	cell[i].force = { fx,fy,fz };
}

// check the total cell number
__global__ void se_verification(int* sum, int* s, int* e, int vnum) {
	const int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= vnum) return;

	if (e[i] >= 0) {
		atomicAdd(sum, e[i] - s[i] + 1);
	}
}

__global__ void movementkernel(Cell* cell, int num) {
	const int i = blockDim.x * blockIdx.x + threadIdx.x;
	if (i >= num) return;
	if (cell[i].sign == false) {
		//printf("movementkernel An unexpected Dead Cell encountered:%d\n", i);
		return;
	}
	float dt = Mechanics_dt;
	float3 pos = cell[i].pos;
	float3 of = cell[i].oldforce;
	float3 f = cell[i].force;
	pos.x += dt / 2.f * (3.f * f.x - of.x);
	pos.y += dt / 2.f * (3.f * f.y - of.y);
	pos.z += dt / 2.f * (3.f * f.z - of.z);

	//printf("Cell %d, F(%f£¬%f£¬%f),OF(%f£¬%f£¬%f),Move(%f£¬%f£¬%f)\n", i, f.x, f.y, f.z, of.x, of.y, of.z, move.x, move.y, move.z);
	//remove the outside cells
	int3 ijk = Find_Mesh(pos,Voxel_length);
	if (!Check_Inside(ijk)) {
		cell[i].sign = false;
		return;
	}
	cell[i].pos.x = pos.x;
	cell[i].pos.y = pos.y;
	cell[i].pos.z = pos.z;
	cell[i].oldforce = f;
	cell[i].force = { 0.f,0.f,0.f };
	cell[i].mesh_idx = Find_Index(ijk);
}

struct MechanicsMesh_struct {
	int* sum = 0;
	int* Mech_Mesh_s = 0;
	int* Mech_Mesh_e = 0;
	int* key = 0;
	const int total_voxel = Voxel_num * Voxel_num * Voxel_num ;

	MechanicsMesh_struct() {
		cudaMalloc(&sum, sizeof(int));
		cudaMalloc(&Mech_Mesh_s, total_voxel * sizeof(int));
		cudaMalloc(&Mech_Mesh_e, total_voxel * sizeof(int));
		cudaMalloc(&key, Max_Cell_num * sizeof(int));
	}

	~MechanicsMesh_struct() {
		cudaFree(sum);
		cudaFree(Mech_Mesh_s);
		cudaFree(Mech_Mesh_e);
		cudaFree(key);
	}

	void FM_update(thrust::device_vector<Cell>& GpuCell, int num) {

		auto GC = thrust::raw_pointer_cast(GpuCell.data());

		Mech_Mesh_se_Reset << <(total_voxel + BlockWidth1d - 1) / BlockWidth1d, BlockWidth1d >> > (Mech_Mesh_s, Mech_Mesh_e, total_voxel);
		updateCellMeshIdx << <(num + BlockWidth1d - 1) / BlockWidth1d, BlockWidth1d >> > (GC, num, key);
		cudaDeviceSynchronize();

		//thrust::sort(GpuCell.begin(), GpuCell.begin() + num);
		thrust::sort_by_key(&key[0], &key[num], GpuCell.begin());
		cudaDeviceSynchronize();
		GC = thrust::raw_pointer_cast(GpuCell.data());

		Mech_Mesh_se_Get << <(num + BlockWidth1d - 1) / BlockWidth1d, BlockWidth1d >> > (GC, num, Mech_Mesh_s, Mech_Mesh_e);
		cudaDeviceSynchronize();

		forcekernel << <(num + BlockWidth1d - 1) / BlockWidth1d, BlockWidth1d >> > (GC, num, Mech_Mesh_s, Mech_Mesh_e);
		cudaDeviceSynchronize();

		movementkernel << <(num + BlockWidth1d - 1) / BlockWidth1d, BlockWidth1d >> > (GC, num);
		cudaDeviceSynchronize();
	}

	void Check_se() {
		int csum = 0;
		cudaMemcpy(sum, &csum, sizeof(int), cudaMemcpyHostToDevice);
		se_verification << <(total_voxel + BlockWidth1d - 1) / BlockWidth1d, BlockWidth1d >> > (sum, Mech_Mesh_s, Mech_Mesh_e, total_voxel);
		cudaDeviceSynchronize();
		cudaMemcpy(&csum, sum, sizeof(int), cudaMemcpyDeviceToHost);
		printf("Check_se:%d\n", csum);
	}
};



