/*
Given a matrix, for any element with value 0,
set all elements from the corresponding row and column on 0, as well.

Implementation using CUDA for NVIDIA GPUs.

@2017 Florin Tulba (florintulba@yahoo.com)
*/

#include "cudaSession.h"

#include <iostream>
#include <cuda_runtime.h>

using namespace std;

CudaSession::CudaSession() {
	cudaDeviceProp props;
	props.asyncEngineCount = 1; // expecting >= 1
	props.canMapHostMemory = 1; // required, as well

	int devId = 0;
	if(cudaChooseDevice(&devId, &props) != cudaSuccess) {
		cerr<<"cudaChooseDevice failed!"<<endl;
		throw runtime_error("cudaChooseDevice failed!");
	}

	if(cudaSetDevice(devId) != cudaSuccess) {
		cerr<<"cudaSetDevice failed!"<<endl;
		throw runtime_error("cudaSetDevice failed!");
	}
	
	if(cudaGetDeviceProperties(&props, devId) != cudaSuccess) {
		cerr<<"cudaGetDeviceProperties failed!"<<endl;
		throw runtime_error("cudaGetDeviceProperties failed!");
	}

	if(props.asyncEngineCount < 1) {
		cerr<<"Current GPU cannot execute kernels and simultaneously perform memory transfers!"<<endl;
		throw runtime_error("asyncEngineCount < 1");
	}

	if(props.canMapHostMemory == 0) {
		cerr<<"Current GPU cannot take advantage of pinned host memory!"<<endl;
		throw runtime_error("canMapHostMemory == 0");
	}
}

CudaSession::~CudaSession() {
	if(cudaDeviceReset() != cudaSuccess)
		cerr<<"cudaDeviceReset failed!"<<endl;
}
