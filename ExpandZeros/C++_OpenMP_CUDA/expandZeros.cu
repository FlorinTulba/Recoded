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
by a block of THREADS_PER_BLOCK threads
*/
__global__ void markZeros(const int * const __restrict__ a,
						  unsigned szA, unsigned n) {
	unsigned pos = blockIdx.x * THREADS_PER_BLOCK + threadIdx.x;
	if(pos >= n) // pos needs to be on the 1st row
		return;

	// deduce foundCols and foundRows from a, szA and n
	bool * const foundCols = const_cast<bool*>((const bool*)a) +
		(ptrdiff_t)nextMultipleOf<256ULL>(size_t(szA) * sizeof(int));
	bool * const foundRows = &foundCols[size_t(n)];

	// Traverse an entire column
	bool shouldMarkCol = false;
	for(; pos < szA; pos += n) {
		if(a[pos] == 0)
			// No matter how unsynchronized the warps in a block are,
			// or if they belong to different blocks,
			// all threads (from that warp) which find a 0 on row floor(pos/n)
			// will write the same 'true' value in foundRows[pos/n]
			foundRows[pos / n] = shouldMarkCol = true;
	}

	// Assigning only if shouldMarkCol is true produces unnecessary divergence.
	// Besides, this spares the initialization of foundCols
	foundCols[pos % n] = shouldMarkCol;
}

void launchMarkZerosKernel(const KernelLaunchConfig &kernelLaunchConfig,
						   const int * const a,
						   unsigned szA, unsigned n) {
	markZeros<<<kernelLaunchConfig.blocksCount(), kernelLaunchConfig.threadsPerBlock(),
				kernelLaunchConfig.sharedMemSize(), kernelLaunchConfig.stream()>>>
		(a, szA, n);
	CHECK_CUDA_KERNEL_LAUNCH(markZeros);
}
