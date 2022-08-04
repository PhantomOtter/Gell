#pragma once

#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <iostream>

#define PI 3.1415927
#define BlockWidth3d 4
#define RAD 1
#define BlockWidth2d 4
#define BlockWidth1d 64
#define Max_Cell_num (2*1000*1000)
#define Cell_Init_num 2347
#define Voxel_num 100//x y z
#define Voxel_length 25 // um
#define Volume_length (Voxel_num*Voxel_length)
#define Sort_Key_Last 3000000

#define Max_Simulation_Time (20.f*24*60)//(10*24*60)
#define Save_data_gap 180 // min
#define Tumor_Proliferation_Rate  1.f/8.5/60.0 //0.04f*1.1/60   //1.f/24.0/60.0

#define O2_Default_Concentration 38.f  // mmHg
#define O2_Diffusion_coef 100000.f // um2/min
#define O2_Decay_rate 0.1f //  /min

#define Tumor_O2_Proliferation_Thres 5.f
#define Tumor_O2_Proliferation_Sat 10.f // mmhg

#define Tumor_Hypoxia_O2_Thres 10.f
#define Tumor_Necrosis_02_Thres 5.f
#define Tumor_Necrosis_02_Max 2.5f
#define Tumor_Necrosis_Rate (1.f / 6.0 / 60.0) // 6hour
#define Tumor_O2_consumption -10.f  //-6.5f;  //per min
#define Dead_O2_consumption_factor 0.1f
#define Tumor_Apoptosis_rate (1.f / 7.0 / 24.0 / 60.0)

#define Biology_dt 6.f // min
#define Mechanics_dt 0.1f   //min
#define Diffusion_dt 0.01f  //min


#define Default_V 2494.f
#define Ccca 0.4f
#define Cccr 10.f
#define Ra_ratio 1.25
#define Tumor_Cell_Radius 8.4f
#define Tumor_CC_Max_Dist 25.f
