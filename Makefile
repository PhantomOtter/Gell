# reference: https://github.com/TravisWThompson1/Makefile_Example_CUDA_CPP_To_Executable

# CUDA directory
CUDA_ROOT_DIR=/usr/local/cuda
CUDA_LIB_DIR= -L$(CUDA_ROOT_DIR)/lib64
CUDA_INC_DIR= -I$(CUDA_ROOT_DIR)/include
CUDA_LINK_LIBS= -lcudart

# Compiler options
CC=g++
CC_FLAGS=
CC_LIBS=
NVCC=nvcc
NVCC_FLAGS=
NVCC_LIBS=

# Project file structure
OBJ_DIR = bin
INC_DIR = include

# Target executable name
EXE = HDS

# Object files
OBJS = $(OBJ_DIR)/Main.o 

# Compile
$(EXE) : $(OBJS)
	$(CC) $(CC_FLAGS) $(OBJS) -o $@ $(CUDA_INC_DIR) $(CUDA_LIB_DIR) $(CUDA_LINK_LIBS)

# Compile main .cpp file to object files:
$(OBJ_DIR)/Main.o : Main.cu
	$(NVCC) $(NVCC_FLAGS)-c $< -o $@ $(NVCC_LIBS)

# Clean objects in object directory
clean:
	$(RM) bin/* *.o $(EXE)
