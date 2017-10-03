/*
Given a matrix, for any element with value 0,
set all elements from the corresponding row and column on 0, as well.

Implementations using OpenMP and CUDA for NVIDIA GPUs.

@2017 Florin Tulba (florintulba@yahoo.com)
*/

#include "cudaSession.h"

using namespace std;

CudaSession::CudaSession() {
	cudaDeviceProp props;
	props.asyncEngineCount = 1; // expecting >= 1
	props.canMapHostMemory = 1; // required, as well

	int devId = 0;
	if(cudaChooseDevice(&devId, &props) != cudaSuccess)
		throw runtime_error("cudaChooseDevice failed!");

	if(cudaSetDevice(devId) != cudaSuccess)
		throw runtime_error("cudaSetDevice failed!");
	
	if(cudaGetDeviceProperties(&props, devId) != cudaSuccess)
		throw runtime_error("cudaGetDeviceProperties failed!");

	if(props.asyncEngineCount < 1)
		throw runtime_error("Current GPU cannot execute kernels and simultaneously perform memory transfers!");

	if(props.canMapHostMemory == 0)
		throw runtime_error("Current GPU cannot take advantage of pinned host memory!");
}

CudaSession::~CudaSession() {
	cudaDeviceReset();
}
