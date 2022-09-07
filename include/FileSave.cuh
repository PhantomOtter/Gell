#pragma once
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/remove.h>
#include <iostream>

#include "Definitions.h"
#include "RandomFunctions.cuh"
#include "BasicFunctions.cuh"
#include "Cells.cuh"
#include "LODkernel.cuh"

// remember to adjust save function when Testkernel is used
//void savecsv_cell(thrust::host_vector<Cell>& C, int currentnum, const std::string f) {
//	const char* filename = f.c_str();
//	FILE* outfile = fopen(filename, "w");
//	fprintf(outfile, "%s,%s,%s,%s,%s,%s,%s,%s,%s, %s,%s,%s\n", "Index", "Sign", "MeshIndex", "Type", "Phase", "x", "y", "z", "r", "V", "Vns", "Vcs");
//	for (int i = 0; i < currentnum; i++) {
//		fprintf(outfile, "%d,%d,%d,%d,%d,%f,%f,%f,%f,%f,%f,%f\n", i, C[i].sign, C[i].mesh_idx, C[i].cell_type, C[i].phase, C[i].pos.x, C[i].pos.y, C[i].pos.z, C[i].r,
//			C[i].V, C[i].Vns, C[i].Vcs);
//	}
//	fclose(outfile);
//}
void savecsv_cell(thrust::host_vector<Cell>& C, int currentnum, const std::string f) {
	const char* filename = f.c_str();
	FILE* outfile = fopen(filename, "w");
	fprintf(outfile, "%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n", "Index", "Sign",   "Phase", "x", "y", "z", "r", "V", "Vns", "Vcs");
	for (int i = 0; i < currentnum; i++) {
		fprintf(outfile, "%d,%d,%d,%f,%f,%f,%f,%f,%f,%f\n", i, C[i].sign, C[i].phase, C[i].pos.x, C[i].pos.y, C[i].pos.z, C[i].r,
			C[i].V, C[i].Vns, C[i].Vcs);
	}
	fclose(outfile);
}

void savecsv_mesh(Mesh_struct* mesh, const std::string f) {
	int idx = 0;
	const char* filename = f.c_str();
	FILE* outfile = fopen(filename, "w");
	fprintf(outfile, "%s,%s,%s,%s\n", "i", "j", "k", "Concentration");
	for (int k = 0; k < Voxel_num; k++) {
		for (int j = 0; j < Voxel_num; j++) {
			for (int i = 0; i < Voxel_num; i++) {
				fprintf(outfile, "%d,%d,%d,%f\n", i, j, k, mesh->cp[idx]);
				idx++;
			}
		}
	}
	fclose(outfile);
}
void savecsv_meshslice(Mesh_struct* mesh, const std::string f) {
	int idx = 0;
	const char* filename = f.c_str();
	FILE* outfile = fopen(filename, "w");
	fprintf(outfile, "%s,%s,%s\n", "x", "y", "Concentration");
	// save the slice where z=0

	for (int j = 0; j < Voxel_num; j++) {
		for (int i = 0; i < Voxel_num; i++) {
			idx = i + j * Voxel_num;
			fprintf(outfile, "%d,%d,%f\n", i, j, mesh->cSlice[idx]);
		}
	}
	fclose(outfile);
}

void savecsv_array(int* cnum, double* time, const std::string f) {
	const int Max_Simulation_Iter = Max_Simulation_Time / Biology_dt;
	const char* filename = f.c_str();
	FILE* outfile = fopen(filename, "w");
	fprintf(outfile, "%s,%s,%s\n", "Index", "Cell Num", "Time");
	for (int i = 0; i <= Max_Simulation_Iter; i++) {
		fprintf(outfile, "%d,%d,%f\n", i, cnum[i], time[i]);
	}
	fclose(outfile);
}

void savecsv_gaparray(int* cnum, double* time, const std::string f) {
	const int Max_Simulation_Iter = Max_Simulation_Time / Biology_dt;
	const char* filename = f.c_str();
	FILE* outfile = fopen(filename, "w");
	fprintf(outfile, "%s,%s,%s\n", "Index", "Cell Num", "Time");
	for (int i = 0; i < Max_Simulation_Iter / Save_data_gap; i++) {
		fprintf(outfile, "%d,%d,%f\n", i, cnum[(i + 1) * Save_data_gap - 1], time[i]);
	}
	fclose(outfile);
}
