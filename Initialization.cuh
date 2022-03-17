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

//void Cell_Initialization(thrust::host_vector<Cell>& C, int N) {
//	float middle = Volume_length / 2;
//	float spacing = 10.f;
//	float radius = 8.f;
//	int cellidx = 0;
//	printf("Volume center is %f, Box length is %f\n", middle, N * spacing);
//	for (int z = 0; z < N; z++) {
//		for (int y = 0; y < N; y++) {
//			for (int x = 0; x < N; x++) {
//				cellidx = Find_Index(x, y, z, N);
//				if (cellidx < C.size()) {
//					float3 p = { middle + spacing * (x - N / 2),
//						middle + spacing * (y - N / 2) ,middle + spacing * (z - N / 2) };
//					if (p.x > 0.f && p.x < Volume_length &&
//						p.y>0.f && p.y < Volume_length &&
//						p.z>0.f && p.z < Volume_length) {
//						C[cellidx].sign = true;
//						C[cellidx].pos = p;
//						C[cellidx].r = radius;
//						C[cellidx].cell_type = 1;
//						C[cellidx].phase = 0;
//					}
//				}
//			}
//		}
//	}
//	printf("Cell initialized\n");
//}


void Cell_sphere_Initialization(thrust::host_vector<Cell>& C, int N) {
	float middle = Volume_length / 2;
	float spacing = 10.f;
	float radius = 8.4f;
	float ballr = radius * pow(N, 1.0 / 3) / 1.612 ;
	printf("Volume center is %f, Cell Init Ball radius is %f", middle, ballr);
	int cellidx = 0;
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
			C[i].cell_type = 1; // tumor
			C[i].phase = 0; // growing
			C[i].V = 2494;
			C[i].Vf = 1871;
			C[i].Vns = 135;
			C[i].Vcs = 489;
			i++;
		}
	}
	printf("\nCell initialized\n");

}