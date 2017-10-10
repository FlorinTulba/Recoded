/*
Given a matrix, for any element with value 0,
set all elements from the corresponding row and column on 0, as well.

Implementations using OpenMP and CUDA for NVIDIA GPUs.

@2017 Florin Tulba (florintulba@yahoo.com)
*/

#include "cudaSession.h"

#include <iostream>

using namespace std;

KernelLaunchConfig::KernelLaunchConfig(unsigned blocksCount_/* = 1U*/,
									   unsigned threadsPerBlock_/* = 32U*/,
									   unsigned shMemSz_/* = 0U*/,
									   cudaStream_t stream_/* = nullptr*/) {
	setBlocksCount(blocksCount_);
	setThreadsPerBlock(threadsPerBlock_);
	setSharedMemSize(shMemSz_);
	setStream(stream_);
}

void KernelLaunchConfig::setBlocksCount(unsigned blocksCount_) {
	if(blocksCount_ == 0U)
		throw invalid_argument(__FUNCTION__ " expects blocksCount_ > 0!");
	_blocksCount = blocksCount_;
}

void KernelLaunchConfig::setThreadsPerBlock(unsigned threadsPerBlock_) {
	if(threadsPerBlock_ == 0U || threadsPerBlock_ % 32U != 0U)
		throw invalid_argument(__FUNCTION__ " expects threadsPerBlock_ to be > 0 and a multiple of 32 !");
	_threadsPerBlock = threadsPerBlock_;
}

void KernelLaunchConfig::setSharedMemSize(unsigned shMemSz_) {
	_shMemSz = shMemSz_;
}

void KernelLaunchConfig::setStream(cudaStream_t stream_) {
	_stream = stream_;
}

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

cudaEvent_t CudaSession::createEvent(unsigned flags/* = cudaEventDefault*/) {
	cudaEvent_t evt;
	CHECK_CUDA_OP(cudaEventCreateWithFlags(&evt, flags));
	reservedEvents.push_back(evt);
	return evt;
}

void CudaSession::releaseDevMem() {
	if(reservedDevMem != nullptr) {
		CHECK_CUDA_OP(cudaFree(reservedDevMem));
		reservedDevMem = nullptr;
	}
}

void CudaSession::destroyStreams() {
	for(const cudaStream_t s : reservedStreams)
		CHECK_CUDA_OP(cudaStreamDestroy(s));

	reservedStreams.clear();
}

void CudaSession::destroyEvents() {
	for(const cudaEvent_t e : reservedEvents)
		CHECK_CUDA_OP(cudaEventDestroy(e));

	reservedEvents.clear();
}

char* CudaSession::getReservedMem() const {
	return (char*)reservedDevMem;
}

const vector<cudaStream_t>& CudaSession::getStreamsPool() const {
	return reservedStreams;
}

const vector<cudaEvent_t>& CudaSession::getEventsPool() const {
	return reservedEvents;
}

CudaSession::~CudaSession() {
	try {
		releaseDevMem();
	} catch(runtime_error&) {}
	try {
		destroyStreams();
	} catch(runtime_error&) {}
	try {
		destroyEvents();
	} catch(runtime_error&) {}
	try {
		CHECK_CUDA_OP(cudaDeviceReset());
	} catch(runtime_error&) {}
}
