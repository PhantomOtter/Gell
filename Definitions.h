#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>

#define PI 3.1415927
#define BlockWidth3d 4
#define RAD 1
#define BlockWidth2d 4
#define BlockWidth1d 64
#define Max_Cell_num 1024*1024*30
#define Max_Prol_num (Max_Cell_num/16)
#define Cell_Init_num 100000
#define Voxel_num 200//x y z
#define Voxel_length 20 // um
#define Volume_length (Voxel_num*Voxel_length)
#define Max_Simulation_Time 10.f*24*60
#define Max_Simulation_Iter 20*24*60*2 //30*24*60
#define Diffusion_Per_Iter 10
#define Save_data_gap 60 // iter

#define Tumor_Proliferation_Rate  1.f/4.0/60.0 //0.04f*1.1/60   //1.f/24.0/60.0

#define O2_Default_Concentration 38.f  // mmHg
#define O2_Diffusion_coef 100000.f // um2/min
#define O2_Decay_rate 0.01f //  /min

#define Tumor_O2_Proliferation_Thres 5.f
#define Tumor_O2_Proliferation_Sat 10.f // mmhg


#define Tumor_Hypoxia_O2_Thres 10.f
#define Tumor_Necrosis_02_Thres 5.f
#define Tumor_Necrosis_02_Max 2.5f
#define Tumor_Necrosis_Rate 1.f/(6.0 * 60.0) // 6hour
#define Tumor_O2_consumption -1.f/2  //-6.5f;  //per min

#define Normal_Hypoxia_O2_Thres 8.f
#define Normal_Necrosis_02_Thres 5.f
#define Normal_02_consumption -10.f  //-6.5f;  //per min

struct Parameters {
	const float Biology_dt = 1;   //min
	const float Diffusion_dt = Biology_dt / Diffusion_Per_Iter;  //min
	//const int dif_per_bio = 10;
	//const int Max_Simulation_Iter = Max_Simulation_Time / Biology_dt;
//	const float O2_Default_Concentration = 38.f;  // mmHg
//	const float O2_Diffusion_coef = 100000.f;   // um2/min
//	const float O2_Decay_rate = 0.01f;   //  /min

	float Normal_Cell_Radius = 8.4f;   // um
	float Normal_Cell_Radius_GR = 0.04f;  //growth rate dr ,unit: min-1
	float Tumor_Cell_Radius = 8.4f;    //um
	float Tumor_Cell_Radius_GR = 0.08f;

	float Vessel_Radius = 10.f;  //um
	float Normal_Cell_Kernel_Radius = 2.f;
	float Tumor_Cell_Kernel_Radius = 2.f;
	float Ccca = 0.4f; // constant for cell cell adhesive f  v*um/min
	float Cccr = 10.f; // constant for cell cell adhesive f  v*um/min
	float R = Tumor_Cell_Radius * 2;
	float Ra_ratio = 1.25;
	float Tumor_CC_Max_Dist = 21.f;  //um  cell-cell interaction max distance


	//float Tumor_Hypoxia_O2_Thres = 0.2f;
	//float Normal_Hypoxia_O2_Thres = 0.2f;
	//float Tumor_Necrosis_02_Thres = 0.1f;
	//float Normal_Necrosis_02_Thres = 0.1f;
	//float Tumor_Proliferation_Rate = 0.1f;
	//float Normal_Proliferation_Rate = 0.05f;

	//float Tumor_Hypoxia_O2_Thres = 10.f;
	//float Normal_Hypoxia_O2_Thres = 10.f;
	//float Tumor_Necrosis_02_Thres = 5.f;
	//float Normal_Necrosis_02_Thres = 5.f;



	//float Normal_02_consumption = -10.f;   //-6.5f;  //per min
	//float Cancer_O2_consumption = -10.f;   //-6.5f;  //per min

	float Tumor_VEGF_ax = Tumor_Hypoxia_O2_Thres;
	float Tumor_VEGF_bx = Tumor_Necrosis_02_Thres;
	float Tumor_VEGF_by = 1.f;

	float Normal_VEGF_ax = Normal_Hypoxia_O2_Thres;
	float Normal_VEGF_bx = Normal_Necrosis_02_Thres;
	float Normal_VEGF_by = 0.5f;
};


