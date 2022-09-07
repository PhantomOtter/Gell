#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>

#define PI 3.1415927
#define RAD 1
#define BlockWidth3d 4
#define BlockWidth2d 4
#define BlockWidth1d 64
#define Max_Cell_num (2*1000*1000)
#define Cell_Init_num 2347
#define Voxel_num 100 //x y z
#define Voxel_length 25 // um
#define Sort_Key_Last -1

#define Max_Simulation_Time (19.f*24*60)
#define Save_data_gap 360 // min

#define O2_Default_Concentration 38.f  // mmHg
#define O2_Diffusion_coef 100000.f // um2 per min
#define O2_Decay_rate 0.1f //  per min
#define Tumor_O2_consumption -10.f  //per min
#define Dead_O2_consumption_factor 0.1f

#define Default_V 2494.f
#define Ccca 0.4f
#define Cccr 10.f
#define Ra_ratio 1.25
#define Tumor_Cell_Radius 8.4f
#define Tumor_CC_Max_Dist 25.f

#define Tumor_O2_Proliferation_Thres 5.f
#define Tumor_O2_Proliferation_Sat 10.f
#define Tumor_Necrosis_02_Thres 5.f
#define Tumor_Necrosis_02_Max 2.5f
#define Tumor_Necrosis_Rate (1.f / 6.0 / 60.0)
#define Tumor_Apoptosis_rate (1.f / 7.0 / 24.0 / 60.0)
#define Tumor_Proliferation_Rate (1.f/8.5/60.0)
#define duration_pre2post (13.f * 60) // premitotic to postmitotic
#define duration_post2q (2.5f * 60) // postmitotic to mature


#define Biology_dt 6.f // min
#define Mechanics_dt 0.1f   //min
#define Diffusion_dt 0.01f  //min