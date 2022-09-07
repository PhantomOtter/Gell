#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <cmath>
#include "Definitions.h"
#include "BasicFunctions.cuh"

struct Cell {
	bool sign = false;
	int mesh_idx = -1;
	float3 pos = { -1.f,-1.f,-1.f };
	float3 force = { 0.f,0.f,0.f };
	float3 oldforce = { 0.f,0.f,0.f };
	float r = -1.f;  // in um
	// total volume
	float V = -1.f;
	// fluid volume
	float Vf = -1.f;
	// cytoplasmic volume
	float Vcs = -1.f;
	// nuclear solid volume
	float Vns = -1.f;

////0 for mature (quiescent)
////1 divide reparation (premitotic)
////2 for growing (postmitotic)
////3 for apoptotic
////4 for early necrotic
////5 for late necrotic

	int phase = -1;
	float cell_clock = 0.f;

	__host__ __device__ void show();
	__host__ __device__ void reset();
	// simplified, edit later | multiplied by cell volume and divided by voxel volume per unit volume um3
	__host__ __device__ float O2_consume(float O2);
	__host__ __device__ float O2_consume();
	__host__ __device__ void Radius_update();
	__host__ __device__ void Volume_update();
	//__host__ __device__ bool Phase_update(float rand, float O2);
	__host__ __device__ bool Phase_update_ki67adv(float rand, float O2);
	//__host__ __device__ bool operator < (const Cell& a) const {
	//	return mesh_idx < a.mesh_idx;
	//}
};

void Cell::show() {
	printf("Cell Idx:%d, Cell pos: %f,%f,%f, Cell force:%f,%f,%f, Cell R:%f, Phase:%d\n", sign, pos.x, pos.y, pos.z, force.x, force.y, force.z,
		r, phase);
}
void Cell::reset() {
	sign = false;
	mesh_idx = -1;
	pos = { -1.f,-1.f,-1.f };
	force = { 0.f,0.f,0.f };
	oldforce = { 0.f,0.f,0.f };

	r = -1.f;
	phase = -1;
	V = -1.f;
	Vf = -1.f;
	Vcs = -1.f;
	Vns = -1.f;
	cell_clock = 0.f;
}

void Cell::Radius_update() {
	r = pow(0.23873241 * V, 1.0 / 3);
}

// return rate /min
float Cell::O2_consume(float O2) {
	float volume_frac = V / Voxel_length / Voxel_length / Voxel_length;
	float rate = 0;

	if (phase>=0 && phase <= 2) {
		rate = Tumor_O2_consumption * volume_frac * O2;
	}
	else if (phase >= 3) {
		rate = Tumor_O2_consumption * volume_frac * O2 * Dead_O2_consumption_factor;
	}
	return rate;
}

float Cell::O2_consume() {
	return Cell::O2_consume(1.f);
}

void Cell::Volume_update() {
	float ff = 0.7502;
	float rf = 3.f / 60.f;  
	float rcs = 0.33f / 60.f;
	float rns = 0.33f / 60.f;
	float Vns_0 = 135.f; // um3
	float Vf_0 = 1871.f;  // 1871 
	float Vcs_0 = 488.f; // standard cytoplasmic:nuclear volume ratio

	////0 for mature (quiescent)
	////1 divide reparation (premitotic)
	////2 for growing (postmitotic)
	////3 for apoptotic
	////4 for early necrotic
	////5 for late necrotic

	if (phase == 0 || phase == 2) {
		// default
	}
	else if (phase == 1) {
		Vns_0 = Vns_0 * 2;
		rcs = 0.27 / 60;
	}
	else if (phase == 3) {
		ff = 0;
		Vns_0 = 0;
		rns = 0.35 / 60;
		rcs = 1.f / 60;
	}
	else if (phase == 4) {
		ff = 1;
		rf = 0.67 / 60;
		Vns_0 = 0;
		rns = 0.013 / 60;
		rcs = 0.0032 / 60;
	}
	else if (phase == 5) {
		ff = 0;
		rf = 0.05 / 60;
		Vns_0 = 0;
		rns = 0.013 / 60;
		rcs = 0.0032 / 60;
	}

	Vf_0 = V * ff;  // 1871 
	Vcs_0 = Vns_0 * 3.615f; // standard cytoplasmic:nuclear volume ratio
	Vf = Vf * (1.f - Biology_dt * rf) + rf * Biology_dt * Vf_0;
	Vns = Vns * (1.f - Biology_dt * rns) + rns * Biology_dt * Vns_0;
	Vcs = Vcs * (1.f - Biology_dt * rcs) + rcs * Biology_dt * Vcs_0;
	V = Vf + Vns + Vcs;
	r = pow(0.23873241 * V, 1.0 / 3);
	return;
}

bool Cell::Phase_update_ki67adv(float rand, float O2) {

	cell_clock = cell_clock + Biology_dt;
	bool proliferate = false;

	float apoptosis_prob = Tumor_Apoptosis_rate * Biology_dt;//0.00319f / 60; // 1.f / 8.6 / 50 / 60; // 2% of cells are in apoptotic phase
	float proliferation_prob = 1.f * (O2 - Tumor_O2_Proliferation_Thres) / (Tumor_O2_Proliferation_Sat - Tumor_O2_Proliferation_Thres) * Tumor_Proliferation_Rate * Biology_dt;  //  quick check : 0.0432f / 60 *20;   ************************************************************ /1.2 - 18 day 1m
	if (O2 > Tumor_O2_Proliferation_Sat) {
		proliferation_prob = Tumor_Proliferation_Rate * Biology_dt;
	}
	else if (O2 < Tumor_O2_Proliferation_Thres) {
		proliferation_prob = 0;
	}

	float necrosis_prob = 1.f * (Tumor_Necrosis_02_Thres - O2) / (Tumor_Necrosis_02_Thres - Tumor_Necrosis_02_Max) * Tumor_Necrosis_Rate * Biology_dt;
	if (O2 > Tumor_Necrosis_02_Thres) {
		necrosis_prob = 0;
	}
	else if (O2 < Tumor_Necrosis_02_Max) {
		necrosis_prob = Tumor_Necrosis_Rate * Biology_dt;
	}

	// float Vcs_stand = 488.f;
	////0 for mature (quiescent)
	////1 divide reparation (premitotic)
	////2 for growing (postmitotic)
	////3 for apoptotic
	////4 for early necrotic
	////5 for late necrotic

	if (phase >= 3) {
		// dead cell
		if (phase == 4 && V >= Default_V * 2) {
			phase = 5;
			cell_clock = 0;
		}

		if (Vcs <= 0.05f * 488) {  // 488 * 0.05 = 24.4
			sign = false;
			cell_clock = 0;
		}
	}
	else{
		// living cell
		// fixed duration process
		if (phase == 1 && cell_clock >= duration_pre2post) {
			phase = 2;
			proliferate = true;
			cell_clock = 0;
		}

		if (phase == 2 && cell_clock >= duration_post2q) {
			phase = 0;
			cell_clock = 0;
		}

		// stochastic process
		if (phase == 0) {
			// to be
			if (rand <= proliferation_prob) {
				phase = 1;
				cell_clock = 0;
			}
			// not to be
			else if (rand <= proliferation_prob + apoptosis_prob) {
				phase = 3;
				cell_clock = 0;
			}
			else if (rand <= proliferation_prob + apoptosis_prob + necrosis_prob) {
				phase = 4;
				cell_clock = 0;
			}
			else {
				// do nothing
			}
		}
	}

	return proliferate;
}