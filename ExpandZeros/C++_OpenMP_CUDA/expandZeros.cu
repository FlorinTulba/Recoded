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

@param a global memory containing the matrix to process.
After the bytes for a, there are 2 regions for foundCols and foundRows.
@param szA the number of elements of `a`
@param n the number of columns of `a`
@param blIdx the index of a batch of consecutive columns to be analyzed within a separate stream
by a block of THREADS_PER_BLOCK threads
*/
__global__ void markZeros(const int * const __restrict__ a,
						  unsigned szA, unsigned n, unsigned blIdx) {
	unsigned pos = blIdx * THREADS_PER_BLOCK + threadIdx.x;
	if(pos >= n) // pos needs to be on the 1st row
		return;

	// deduce foundCols and foundRows from a, szA and n
	bool * const foundCols = const_cast<bool*>(reinterpret_cast<const bool*>(a)) +
		(ptrdiff_t)nextMultipleOf<256ULL>(size_t(szA) * sizeof(int));
	bool * const foundRows = &foundCols[nextMultipleOf<256ULL>(size_t(n))];

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
						   unsigned szA, unsigned n, unsigned blIdx) {
	markZeros<<<kernelLaunchConfig.blocksCount, kernelLaunchConfig.threadsPerBlock,
				kernelLaunchConfig.shMemSz, kernelLaunchConfig.stream>>>
		(a, szA, n, blIdx);
	CHECK_CUDA_KERNEL_LAUNCH(markZeros);
}
