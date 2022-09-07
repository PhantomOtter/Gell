#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/remove.h>
#include <iostream>

#include "RandomFunctions.cuh"
#include "BasicFunctions.cuh"
#include "Cells.cuh"



void Cell_sphere_Initialization(thrust::host_vector<Cell>& C, int N) {
	float middle = Voxel_num * Voxel_length / 2;
	float spacing = 10.f;
	float radius = Tumor_Cell_Radius;
	float ballr = radius * pow(N, 1.0 / 3) / 1.612 * 1.87;
	printf("Volume center is %f, Cell Init Ball radius is %f", middle, ballr);
	float x, y, z;
	for (int i = 0; i < N;) {
		x = 2.f * ((float)rand()) / RAND_MAX - 1.f;
		y = 2.f * ((float)rand()) / RAND_MAX - 1.f;
		z = 2.f * ((float)rand()) / RAND_MAX - 1.f;
		//printf("%f %f %f ", x, y, z);
		if (x * x + y * y + z * z < 1) {
			C[i].sign = true;
			C[i].pos = { middle + x * ballr,middle + y * ballr,middle + z * ballr };
			C[i].force = { 0.f,0.f,0.f };
			C[i].oldforce = { 0.f,0.f,0.f };
			C[i].r = radius;
			C[i].phase = 0; // quiescent
			C[i].V = Default_V;
			C[i].Vf = 1871;
			C[i].Vns = 135;
			C[i].Vcs = 489;
			C[i].mesh_idx = Find_Index(Find_Mesh(C[i].pos,Voxel_length),Voxel_num);
			i++;
		}
	}
	printf("\nCell initialized\n");

}