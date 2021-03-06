
#include "cuda_runtime.h"
#include "device_launch_parameters.h" 
#include <thrust/host_vector.h>
#include <thrust/device_vector.h>
#include <thrust/remove.h>
#include <thrust/copy.h>
#include <thrust/sort.h>
#include <thrust/fill.h>
#include <thrust/functional.h>
#include <iostream>
#include <string>
#include <time.h>

#include "Definitions.h"
#include "RandomFunctions.cuh"
#include "BasicFunctions.cuh"
#include "Cells.cuh"
#include "Mesh.cuh"

#include "Initialization.cuh"
#include "FileSave.cuh"
#include "PDkernel.cuh"
#include "LODkernel.cuh"
#include "FMkernel.cuh"

#include "Testkernel.cuh"


#define show_time_every_iter   false  // if false, only show every time after Save_data_gap
#define save_intermediate_Cell false
#define save_intermediate_Mesh false

// Gell version based on Thrust

int main(void)
{
	showGPUinfo(true);

	clock_t Initstart = clock();
	std::string datapath = "D:\\Gell_Data\\Spheroid2\\Speed\\";

	// Simu time record
	double iterduration = 0;
	double gapduration = 0;
	int cellnumrecord[Max_Simulation_Iter] = { 0 };
	double gapsimutimerecord[Max_Simulation_Iter / Save_data_gap] = { 0 };  // simulation time without saving time
	double simutimerecord[Max_Simulation_Iter] = { 0 };                     // simulation time without saving time
	int filesaveidx = 0;        // index of how may Save_data_gap(currently 60 min) of simulation has passed 
	bool showandsave = false;   // weather Save_data_gap condition matched

	// para struct init
	Parameters cPara;
	Parameters* gPara = 0;
	cudaMalloc(&gPara, sizeof(Parameters));
	cudaMemcpy(gPara, &cPara, sizeof(Parameters), cudaMemcpyHostToDevice);


	// mesh struct init
	const dim3 blockSize2d(BlockWidth2d, BlockWidth2d);
	const int gdim2d = (Voxel_num + BlockWidth2d - 1) / BlockWidth2d;
	const dim3 gridSize2d(gdim2d, gdim2d);
	Mesh_struct O2Mesh(O2_Default_Concentration, cPara.Diffusion_dt, O2_Diffusion_coef, O2_Decay_rate);
	savecsv_meshslice(&O2Mesh, datapath + "Mesh_slice_0.csv");
	// savecsv_mesh(&O2Mesh, datapath + "02_Mesh_0.csv");

	// Random num init
	curandState* curand_states;
	cudaMalloc(&curand_states, sizeof(curandState) * Max_Cell_num);
	set_random_states << <(Max_Cell_num + BlockWidth1d - 1) / BlockWidth1d, BlockWidth1d >> > (curand_states);
	cudaDeviceSynchronize();
	clock_t Initend = clock();
	std::cout << "Initialization Finished, takes " << (double)(Initend - Initstart) / CLOCKS_PER_SEC << std::endl;

	// Cell vector init
	srand(12345);
	int initnum = Cell_Init_num;
	int currentnum = initnum;
	int* newnum = 0;
	int cnewnum = 0;
	cudaMalloc(&newnum, sizeof(int));
	cudaMemset(newnum, 0, sizeof(int));
	thrust::host_vector<Cell> CpuCell(Max_Cell_num);
	//Cell_Initialization(CpuCell, initnum);
	Cell_sphere_Initialization(CpuCell, initnum);
	savecsv_cell(CpuCell, currentnum, datapath + "Gell_0.csv");
	thrust::device_vector<Cell> GpuCell(Max_Cell_num);
	thrust::device_vector<Cell> TempCell(Max_Cell_num / 16);
	GpuCell = CpuCell;
	Cell* GC = thrust::raw_pointer_cast(GpuCell.data());
	Cell* TC = thrust::raw_pointer_cast(TempCell.data());



	// record the idx of first cell inside each cell mesh, last additional member is current cell num
	const int CellMeshArrayLen = Voxel_num * Voxel_num * Voxel_num + 1;
	thrust::device_vector<int> CellIndex_of_StartCell_of_CellMesh(CellMeshArrayLen);
	//if cell[i].cellmeshidx - cell[i - 1].cellmeshidx > 0, then i is a start index this cell mesh, record the cellmeshidx of i in the array
	//thrust::device_vector<int> CellIndex_of_StartCell(initnum * initnum * initnum);
	int* pointerA = 0;
	int* pointerB = 0;





	//Simu core
	clock_t Simustart = clock();
	//currentnum = GpuCell.size();
	std::cout << "Simulation Start -- Current Cell Num: " << currentnum << std::endl;
	std::cout << std::endl;
	clock_t showsimutimestart = clock();
	clock_t showsimutimeend = clock();
	float current_cell_time = 0.f;
	int SimuIter = 0;
	for (float current_cell_time = 0.f; current_cell_time <= Max_Simulation_Time; current_cell_time += cPara.Biology_dt) {
		SimuIter++;
		clock_t Iterstart = clock();
		//Birth Module
		currentnum = CellBirth_kernel(O2Mesh.p, GpuCell, TempCell, gPara, curand_states, currentnum, newnum);
		//Force and Movement Module
		Cellmesh_FM_kernel(GpuCell, CellIndex_of_StartCell_of_CellMesh, gPara, currentnum);
		//Death Module
		currentnum = CellDeath_kernel(GpuCell, gPara, curand_states, currentnum);

		//Diffusion Decay Uptake Release Module
	
		GC = thrust::raw_pointer_cast(GpuCell.data());
		for (int diff = 0; diff < Diffusion_Per_Iter; diff++) {
			meshupdatekernel << < (currentnum + BlockWidth1d - 1) / BlockWidth1d, BlockWidth1d >> > (O2Mesh.p, GC, gPara, currentnum);
			cudaDeviceSynchronize();
			LODsolver(O2Mesh.p, O2Mesh.E, O2Mesh.F, gPara);
			Setboundary << <gridSize2d, blockSize2d >> > (O2Mesh.p, O2_Default_Concentration);
			cudaDeviceSynchronize();
		}



		clock_t Iterend = clock();
		iterduration = (double)(Iterend - Iterstart) / CLOCKS_PER_SEC;
		gapduration += iterduration;
		cellnumrecord[SimuIter] = currentnum;
		simutimerecord[SimuIter] = iterduration;

		//**************************** Save Show Part ****************************//

		showandsave = ((SimuIter + 1) % Save_data_gap == 0);

		if (show_time_every_iter) {
			std::cout << "Simulation Iter " << SimuIter << " Finished, takes " << iterduration << "s" << std::endl;
			std::cout << "Cell Num: " << currentnum << std::endl;
			std::cout << "Accumulative Time " << (double)(Iterend - Simustart) / CLOCKS_PER_SEC << "s" << std::endl;
			std::cout << std::endl;
		}
		if (showandsave) {
			filesaveidx = (SimuIter + 1) / Save_data_gap;
			gapsimutimerecord[filesaveidx - 1] = gapduration;

			// test module monitor the cloest neighbour of each individual cell
			// time cost not included in the simulation time

			//findcloest << <(currentnum + BlockWidth1d - 1) / BlockWidth1d, BlockWidth1d >> > (GC, gcp, gsp, currentnum);
			//cudaDeviceSynchronize();


			if (save_intermediate_Cell) {
				try {
					//std::cout << "CpuCell size: " << CpuCell.size() << ",GpuCell size: " << GpuCell.size() << std::endl;
					CpuCell = GpuCell;
				}
				catch (thrust::system_error& e) {
					std::cerr << "Save Module Copy back Error: " << e.what() << std::endl;
					exit(-1);
				}
				catch (std::bad_alloc& e) {
					std::cerr << "Save Module Copy back Error: " << e.what() << std::endl;
					exit(-1);
				}
				savecsv_cell(CpuCell, currentnum, datapath + "Gell_" + std::to_string(filesaveidx) + ".csv");
			}
			if (save_intermediate_Mesh) {
				// whole mesh data
				//O2Mesh.move_to_CPU();
				// only a slice
				O2Mesh.Slice_to_CPU();
				// savecsv_mesh(&O2Mesh, datapath + "Mesh_" + std::to_string(filesaveidx) + ".csv");
				savecsv_meshslice(&O2Mesh, datapath + "Mesh_slice_" + std::to_string(filesaveidx) + ".csv");
			}
			showsimutimeend = clock();
			showGPUinfo(true);
			std::cout << "Simulation from " << filesaveidx - 1 << " to " << filesaveidx << " * " << Save_data_gap << " iter finished" << std::endl;
			std::cout << "Time " << (int)((filesaveidx - 1) * Save_data_gap * cPara.Biology_dt / 60) << " Hour" << std::endl;
			std::cout << "Current Cell Num is " << currentnum << std::endl;
			std::cout << "( " << currentnum / 1000 << " K | " << (float)currentnum / 1000 / 1000 << " M )" << std::endl;
			std::cout << "Simulation Time consumption is " << gapduration << "s" << std::endl;
			std::cout << "Simu & Save Time consumption is " << (double)(showsimutimeend - showsimutimestart) / CLOCKS_PER_SEC << "s" << std::endl;
			std::cout << "Accumulative Time " << (double)(showsimutimeend - Simustart) / CLOCKS_PER_SEC << "s" << std::endl;
			std::cout << std::endl;
			gapduration = 0;
			showsimutimestart = clock();
		}
		//**************************** Save Show Part ****************************//
	}

	clock_t Simuend = clock();
	std::cout << std::endl;
	std::cout << "Simulation Finished, takes " << (double)(Simuend - Simustart) / CLOCKS_PER_SEC << std::endl;

	// Save final CSV if no intermediate info is saved 
	if (!save_intermediate_Cell) {
		clock_t Savestart = clock();
		if (!save_intermediate_Cell) {
			CpuCell = GpuCell;
			savecsv_cell(CpuCell, currentnum, datapath + "Gell_Simuend.csv");
		}
		if (!save_intermediate_Mesh) {
			O2Mesh.move_to_CPU();
			O2Mesh.Slice_to_CPU();
			savecsv_mesh(&O2Mesh, datapath + "Mesh_Simuend.csv");
			savecsv_meshslice(&O2Mesh, datapath + "Mesh_slice_Simuend.csv");
		}
		clock_t Saveend = clock();
		std::cout << "Final File Save Finished, takes " << (double)(Saveend - Savestart) / CLOCKS_PER_SEC << std::endl;
	}

	//thrust::host_vector<int> cpu_CellIndex_of_StartCell_of_CellMesh = CellIndex_of_StartCell_of_CellMesh;
	//savecsv_StartCell_array(cpu_CellIndex_of_StartCell_of_CellMesh, datapath + "Gell_CellMesh_StartCell_record.csv");
	savecsv_array(cellnumrecord, simutimerecord, datapath + "Gell_Time_record.csv");
	savecsv_gaparray(cellnumrecord, gapsimutimerecord, datapath + "Gell_GapTime_record_1.csv");

	std::cout << "Whole Simulation and Saving process finished!" << std::endl;

	// Free Spacec
	cudaFree(gPara);
	cudaFree(curand_states);
	cudaFree(newnum);

	return 0;
}
