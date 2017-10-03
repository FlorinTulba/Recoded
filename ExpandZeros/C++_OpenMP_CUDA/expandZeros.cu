/*
Given a matrix, for any element with value 0,
set all elements from the corresponding row and column on 0, as well.

Implementations using OpenMP and CUDA for NVIDIA GPUs.

@2017 Florin Tulba (florintulba@yahoo.com)
*/

#include "expandZerosCUDA.h"

using namespace std;

/**
Kernel marking with true the rows / columns containing zeros

The input data is accessed directly from the global device memory in a coalesced fashion.
The 2 output arrays `foundRows` and `foundCols` might marginally benefit from
computing them first in shared memory and copying them afterwards to global memory. 

@param a global memory containing the matrix to process
@param szA the number of elements of `a`
@param n the number of columns of `a`
@param blIdx the index of a batch of consecutive columns to be analyzed within a separate stream
by a block of THREADS_PER_BLOCK threads
@param foundRows reports the rows containing values equal to 0
@param foundCols reports the columns containing values equal to 0
*/
__global__ void markZeros(const int * const __restrict__ a,
						  unsigned szA, unsigned n, unsigned blIdx,
						  bool * const __restrict__ foundRows,
						  bool * const __restrict__ foundCols) {
	unsigned pos = blIdx * THREADS_PER_BLOCK + threadIdx.x;
	if(pos >= n) // pos needs to be on the 1st row
		return;

	// Traverse an entire column
	bool shouldMarkCol = false;
	for(; pos < szA; pos += n) {
		if(a[pos] == 0)
			foundRows[pos / n] = shouldMarkCol = true;
	}

	if(shouldMarkCol)
		foundCols[pos % n] = true;
}

void launchMarkZerosKernel(const KernelLaunchConfig &kernelLaunchConfig,
						   const int * const a,
						   unsigned szA, unsigned n, unsigned blIdx,
						   bool * const foundRows, bool * const foundCols) {
	markZeros<<<kernelLaunchConfig.blocksCount, kernelLaunchConfig.threadsPerBlock,
				kernelLaunchConfig.shMemSz, kernelLaunchConfig.stream>>>
		(a, szA, n, blIdx, foundRows, foundCols);
	CHECK_CUDA_KERNEL_LAUNCH(markZeros);
}
