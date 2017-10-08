/*
Given a matrix, for any element with value 0,
set all elements from the corresponding row and column on 0, as well.

Implementations using OpenMP and CUDA for NVIDIA GPUs.

@2017 Florin Tulba (florintulba@yahoo.com)
*/

#include "cudaSession.h"

#include <iostream>

using namespace std;

CudaSession::CudaSession() {
	cudaDeviceProp props;
	props.asyncEngineCount = 1; // expecting >= 1
	props.canMapHostMemory = 1; // required, as well

	int devId = 0;

	CHECK_CUDA_OP(cudaChooseDevice(&devId, &props));
	CHECK_CUDA_OP(cudaSetDevice(devId));
	CHECK_CUDA_OP(cudaGetDeviceProperties(&props, devId));

	if(props.asyncEngineCount < 1)
		throw runtime_error("Current GPU cannot execute kernels and simultaneously perform memory transfers!");

	if(props.canMapHostMemory == 0)
		throw runtime_error("Current GPU cannot take advantage of pinned host memory!");
}

void* CudaSession::reserveDevMem(size_t sz) {
	if(reservedDevMem != nullptr)
		throw logic_error("Please release the device memory explicitly before reserving anew!");

	CHECK_CUDA_OP(cudaMalloc(&reservedDevMem, sz));
	return reservedDevMem;
}

cudaStream_t CudaSession::createStream(unsigned flags/* = cudaStreamDefault*/, int priority/* = 0*/) {
	cudaStream_t newStream;
	CHECK_CUDA_OP(cudaStreamCreateWithPriority(&newStream, flags, priority));
	reservedStreams.push_back(newStream);
	return newStream;
}

void CudaSession::destroyStreams() {
	for(const cudaStream_t s : reservedStreams)
		CHECK_CUDA_OP(cudaStreamDestroy(s));

	reservedStreams.clear();
}

void CudaSession::releaseDevMem() {
	if(reservedDevMem != nullptr) {
		CHECK_CUDA_OP(cudaFree(reservedDevMem));
		reservedDevMem = nullptr;
	}
}

const vector<cudaStream_t>& CudaSession::getReservedStreams() const {
	return reservedStreams;
}

char* CudaSession::getReservedMem() const {
	return (char*)reservedDevMem;
}

CudaSession::~CudaSession() {
	releaseDevMem();
	destroyStreams();
	CHECK_CUDA_OP(cudaDeviceReset());
}
