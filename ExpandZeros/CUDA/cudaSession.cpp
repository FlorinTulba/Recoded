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
	if(cudaSetDevice(0) != cudaSuccess)
		cerr<<"cudaSetDevice failed!  Do you have a CUDA-capable GPU installed?"<<endl;
}

CudaSession::~CudaSession() {
	if(cudaDeviceReset() != cudaSuccess)
		cerr<<"cudaDeviceReset failed!"<<endl;
}
