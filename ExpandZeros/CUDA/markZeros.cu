/*
Given a matrix, for any element with value 0,
set all elements from the corresponding row and column on 0, as well.

Implementation using CUDA for NVIDIA GPUs.

@2017 Florin Tulba (florintulba@yahoo.com)
*/

#include "markZeros.h"

#include <device_launch_parameters.h>

#include <iostream>

using namespace std;

enum { THREADS_PER_BLOCK = 128 };

__global__ void markZeros(const int * const __restrict__ a, unsigned szA, unsigned n,
						  bool * const __restrict__ foundRows,
						  bool * const __restrict__ foundCols) {
	unsigned pos = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
	if(pos >= n) // pos needs to be on the 1st row
		return;

	bool shouldMarkCol = false;
	for(unsigned row = 0U; pos < szA; pos += n, ++row) {
		if(a[pos] == 0)
			foundRows[row] = shouldMarkCol = true;
	}

	if(shouldMarkCol)
		foundCols[pos % n] = true;
}

cudaError_t findZeros(const int *a, unsigned m, unsigned n,
					  bool *foundRows, bool *foundCols) {
	const unsigned szA = m * n,
		blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
	int *devA = nullptr;
	bool *devFoundRows = nullptr, *devFoundCols = nullptr;
	cudaError_t cudaStatus = cudaSuccess;

#define CHECK_CUDA_OP(op) \
	cudaStatus = op; \
	if(cudaStatus != cudaSuccess) { \
		cerr<<"Operation " #op " failed because: "<<cudaGetErrorString(cudaStatus)<<endl; \
		goto Leaving; \
	}


	CHECK_CUDA_OP(cudaMalloc((void**)&devA, szA * sizeof(int)));
	CHECK_CUDA_OP(cudaMalloc((void**)&devFoundRows, m * sizeof(bool)));
	CHECK_CUDA_OP(cudaMalloc((void**)&devFoundCols, n * sizeof(bool)));

	CHECK_CUDA_OP(cudaMemcpy((void*)devA, a, szA * sizeof(int), cudaMemcpyHostToDevice));

	markZeros<<<blocks, THREADS_PER_BLOCK>>>
		(devA, szA, n, devFoundRows, devFoundCols);
	CHECK_CUDA_OP(cudaGetLastError());

	//CHECK_CUDA_OP(cudaDeviceSynchronize());

	CHECK_CUDA_OP(cudaMemcpy((void*)foundRows, (void*)devFoundRows, m * sizeof(bool),
		cudaMemcpyDeviceToHost));
	CHECK_CUDA_OP(cudaMemcpy((void*)foundCols, (void*)devFoundCols, n * sizeof(bool),
		cudaMemcpyDeviceToHost));

Leaving:
	cudaFree(devFoundCols);
	cudaFree(devFoundRows);
	cudaFree(devA);
    
    return cudaStatus;

#undef CHECK_CUDA_OP
}
