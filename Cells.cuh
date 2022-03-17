#pragma once
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>
#include <cmath>

#include "Definitions.h"
#include "BasicFunctions.cuh"

struct Cell {
	bool sign = false;
	float3 pos = { -1.f,-1.f,-1.f };
	int posmeshidx = -1;
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
	//first num
	//0 for normal; 
	//1 for cancer; 

	//0 for growing
	//1 for mature
	//2 for hypoxic

	//3 for apoptotic
	//4 for early necrotic
	//5 for late necrotic

	//2 for endothelial

	int cell_type = -1;
	int phase = -1;
	int type_index = -1;


	__host__ __device__ void show();
	__host__ __device__ void reset();
	// simplified, edit later | multiplied by cell volume and divided by voxel volume per unit volume um3
	__host__ __device__ float O2_consume(Parameters* gPara, float O2);
	__host__ __device__ float VEGF_release(Parameters* gPara, float O2);
	__host__ __device__ void Radius_update();
	__host__ __device__ void Volume_update(float dt);
	__host__ __device__ bool Phase_update(float rand,float O2,float dt);
	__host__ __device__ bool operator < (const Cell& a) const {
		return posmeshidx < a.posmeshidx;
	}
};

void Cell::show() {
	printf("Cell Idx:%d, Cell pos: %f,%f,%f, Cell force:%f,%f,%f, Cell R:%f, Cell type:%d\n", sign, pos.x, pos.y, pos.z, force.x, force.y, force.z,
		r, cell_type);
}
void Cell::reset() {
	sign = false;
	pos = { -1.f,-1.f,-1.f };
	force = { 0.f,0.f,0.f };
	oldforce = { 0.f,0.f,0.f };

	r = -1.f;
	cell_type = -1;
	phase = -1;
	type_index = -1;
	V = -1.f;
	Vf = -1.f;
	Vcs = -1.f;
	Vns = -1.f;
}

void Cell::Radius_update() {
	r = pow(0.23873241 * V, 1.0 / 3);
}
// return rate /min
float Cell::O2_consume(Parameters* gPara, float O2) {

	float volume_frac = V / Voxel_length / Voxel_length / Voxel_length;
	switch (cell_type) {
	case 0:
		return Normal_02_consumption * volume_frac * O2;
	case 1:
		switch (phase) {
		case 0:
		case 1:
			return Tumor_O2_consumption * volume_frac * O2;
			break;
		case 2: // hypoxic
			return Tumor_O2_consumption * volume_frac * O2;
			break;
		case 3:
		case 4:
		case 5:
		default:
			return 0;
			break;
		}
	case 2:
		return 0.f;
	default:
		return 0.f;
	}
}
// return rate
float Cell::VEGF_release(Parameters* gPara, float O2) {
	float volume_frac = V / Voxel_length / Voxel_length / Voxel_length;
	switch (cell_type) {
	case 0:
		return Linear_step(gPara->Normal_VEGF_ax, 0.f, gPara->Normal_VEGF_bx, gPara->Normal_VEGF_by, O2) * volume_frac;
	case 1:
		return Linear_step(gPara->Tumor_VEGF_ax, 0.f, gPara->Tumor_VEGF_bx, gPara->Tumor_VEGF_by, O2) * volume_frac;
	default:
		return 0.f;
	}
}
void Cell::Volume_update(float dt) {
	float ff = 0.7502;
	// standard cytoplasmic:nuclear volume ratio
	float fcn = 3.615;
	float rf = 3.f / 60.f;  // per min
	//  9 hour s phase
	float rcs = 0.33f / 60.f;
	float rns = 0.33f / 60.f;
	float Vf_0 = V * ff;  // 1871 
	float Vns_0 = 135.f; // um3
	float Vcs_0 = fcn * Vns_0;  // 488
	float V_0 = 2494.f;  // um3
	switch (cell_type) {
	// normal cell
	case 0:
		break;
	// tumor cell
	case 1:
		//0 for growing
		//1 for mature
		//2 for hypoxic
		//3 for apoptotic
		//4 for early necrotic
		//5 for late necrotic
		//6 for divide prepare
		switch (phase) {
			// tumor cell growing
		case 0:
			// standard water fraction
			ff = 0.7502;
			// standard cytoplasmic:nuclear volume ratio
			fcn = 3.615;
			rf = 3.f / 60.f;  // per min
			//  9 hour s phase
			rcs = 0.33f / 60.f;
			rns = 0.33f / 60.f;
			Vf_0 = V * ff;  // 1871 
			Vns_0 = 135.f; // um3
			Vcs_0 = fcn * Vns_0;  // 488
			V_0 = 2494.f;  // um3
			Vf = Vf * (1.f - dt * rf) + rf * dt * Vf_0;
			Vns = Vns * (1.f - dt * rns) + rns  * dt * Vns_0;
			Vcs = Vcs * (1.f - dt * rcs) + rcs  * dt * Vcs_0;
			V = Vf + Vns + Vcs;
			Radius_update();
			if (V / V_0 >= 0.95) {
				phase = 1;
				V = 2494;
				Vf = 1871;
				Vcs = 488;
				Vns = 135;
				Radius_update();
			}
			break;
			//tumor cell mature
		// mature
		case 1:
			break;
		// hypoxic
		case 2:
			break;
		// apoptotic
		case 3:
			// standard water fraction
			ff = 0;
			// standard cytoplasmic:nuclear volume ratio
			fcn = 0;
			rf = 3.f / 60.f;  // per min
			// 3 hour cytoplasmic blebbing
			rcs = 1.f / 60.f;
			// 8.6 hour nucleus degradation
			rns = 0.35f / 60.f;
			// water loss
			Vf_0 = V * ff;  // 0
			Vns_0 = 0.f; // um3
			Vcs_0 = fcn * Vns_0;  // 488
			Vf = Vf * (1.f - dt * rf) + rf * dt * Vf_0;
			Vns = Vns * (1.f - dt * rns) + rns * dt * Vns_0;
			Vcs = Vcs * (1.f - dt * rcs) + rcs * dt * Vcs_0;
			V = Vf + Vns + Vcs;
			Radius_update();
			if (Vns / 135.f <= 0.05) {
				reset();
			}
			break;
		//tumor cell necrosis 1 oncisis
		case 4:
			// cell swelling
			ff = 1;
			fcn = 0;
			// 6 hour rupture
			rf = 0.67f / 60.f;  
			// phospholipid half life 100-300 hour
			rcs = 0.0032f / 60.f;
			// DNA half life 54 hour
			rns = 0.013f / 60.f;
			Vf_0 = V * ff;  // V
			Vns_0 = 0.f; // 0
			Vcs_0 = fcn * Vns_0;  // 0
			V_0 = 2*2494;  // rupture
			Vf = Vf * (1.f - dt * rf) + rf * dt * Vf_0;
			Vns = Vns * (1.f - dt * rns) + rns * dt * Vns_0;
			Vcs = Vcs * (1.f - dt * rcs) + rcs * dt * Vcs_0;
			V = Vf + Vns + Vcs;
			Radius_update();
			if (V >= V_0) {
				phase = 5;
			}
			break;
		//tumor cell necrosis 2 late necrosis
		case 5:
			// rupture
			ff = 0;
			fcn = 0;
			// 60 hour
			rf = 0.05f / 60.f; 
			rcs = 0.0032f / 60.f;
			rns = 0.013f / 60.f;
			Vf_0 = V * ff;  // V
			Vns_0 = 0.f; // 0
			Vcs_0 = fcn * Vns_0;  // 0
			Vf = Vf * (1.f - dt * rf) + rf * dt * Vf_0;
			Vns = Vns * (1.f - dt * rns) + rns * dt * Vns_0;
			Vcs = Vcs * (1.f - dt * rcs) + rcs * dt * Vcs_0;
			V = Vf + Vns + Vcs;
			Radius_update();
			if (Vcs / 488 <= 0.05) {
				reset();
			}
			break;
		case 6:
			// standard water fraction
			ff = 0.7502;
			// standard cytoplasmic:nuclear volume ratio
			fcn = 3.615;
			rf = 3.f / 60.f;  // per min
			//  rate of cytoplasmic solid biomass creation
			rcs = 0.27f / 60.f;
			// rate of nuclear biomass creation
			rns = 0.33f / 60.f;
			Vf_0 = V * ff;  // 1871 
			Vns_0 = 135.f*2; // um3
			Vcs_0 = fcn * Vns_0;  // 488
			V_0 = 2494.f;  // um3
			Vf = Vf * (1.f - dt * rf) + rf * dt * Vf_0;
			Vns = Vns * (1.f - dt * rns) + rns * dt * Vns_0;
			Vcs = Vcs * (1.f - dt * rcs) + rcs * dt * Vcs_0;
			V = Vf + Vns + Vcs;
			Radius_update();
			break;
		default:
			break;
		}
		break;
	// endothelial cell
	case 2:
		return;
	default:
		break;;
	}
}



// no proliferation
bool Cell::Phase_update(float rand,float O2,float dt) {
	bool proliferate = false;
	float hypoxia_th = 0.2f;
	float apoptotic_prob = 1.0 / (7.0 * 24.0 * 60.0) * dt;//0.00319f / 60; // 1.f / 8.6 / 50 / 60; // 2% of cells are in apoptotic phase

	float proliferation_prob = 1.f * (O2 - Tumor_O2_Proliferation_Thres) / (Tumor_O2_Proliferation_Sat - Tumor_O2_Proliferation_Thres) * Tumor_Proliferation_Rate*dt;  //  quick check : 0.0432f / 60 *20;   ************************************************************ /1.2 - 18 day 1m
	if (O2 > Tumor_O2_Proliferation_Sat) {
		proliferation_prob = Tumor_Proliferation_Rate * dt;
	}
	else if (O2 < Tumor_O2_Proliferation_Thres) {
		proliferation_prob = 0;
	}


	float necrosis_th = Tumor_Necrosis_02_Thres;

	float necrosis_prob = 1.f *(Tumor_Necrosis_02_Thres - O2)/(Tumor_Necrosis_02_Thres - Tumor_Necrosis_02_Max)* Tumor_Necrosis_Rate * dt;//(1.f - O2 / necrosis_th) / 100.f;
	if (O2 > Tumor_Necrosis_02_Thres) {
		necrosis_prob = 0;
	}else if(O2 < Tumor_Necrosis_02_Max){
		necrosis_prob = Tumor_Necrosis_Rate * dt;
	}

	float Vcs_stand = 488.f;
	switch (cell_type) {
	case 0:
		break;
	case 1:
		switch (phase) {
		//0 for growing
		//1 for mature
		//2 for hypoxic
		//3 for apoptotic
		//4 for early necrotic
		//5 for late necrotic
		//6 divide reparation
		case 0:
			// go mature (in volume update)

			// go hypoxic, necrotic
			if (O2 <= hypoxia_th) {
				phase = 2;
			}
			// go apoptotic
			if (rand <= apoptotic_prob) {
				phase = 3;
				break;
			}

			break;
			
		case 1:

			// go hypoxic, necrotic
			if (O2 <= hypoxia_th) {
				phase = 2;
			}

			if (rand <= apoptotic_prob) {
				phase = 3;
				break;
			}

			if (O2 < Tumor_Necrosis_02_Thres && rand <= apoptotic_prob + necrosis_prob) {
				phase = 4;
				break;
			}
			else if (O2 > Tumor_O2_Proliferation_Thres && rand <= apoptotic_prob + proliferation_prob) {
				// entry divide reparation
				phase = 6;
				break;
			}
			break;
		case 2:
			if (O2 >= hypoxia_th) {
				phase = 0;
			}

			if (rand <= apoptotic_prob) {
				phase = 3;
				break;
			}


			if (O2 < Tumor_Necrosis_02_Thres && rand <= apoptotic_prob + necrosis_prob) {
				phase = 4;
				break;
			}
			else if (O2 > Tumor_O2_Proliferation_Thres && rand <= apoptotic_prob + proliferation_prob) {
				// entry divide reparation
				phase = 6;
				break;
			}

			break;
		case 3:
			// no change
			break;
		case 4:
			// no change
			break;
		case 5:
			// no change
			break;
		case 6:
			if (rand <= apoptotic_prob) {
				phase = 3;
				break;
			}

			//if (O2 <= hypoxia_th) {
			//	phase = 2;
			//}

			else if (Vcs / Vcs_stand >= 1.95) {
				proliferate = true;
				break;
			}
			break;
		default:
			break;
		}
		break;
	case 2:
		break;
	}
	return proliferate;
}