/*
Given a matrix, for any element with value 0,
set all elements from the corresponding row and column on 0, as well.

Implementation using CUDA for NVIDIA GPUs.

@2017 Florin Tulba (florintulba@yahoo.com)
*/

#include "expandZeros.h"
#include "cudaSession.h"

#include <device_launch_parameters.h>

#include <iostream>

using namespace std;

enum { THREADS_PER_BLOCK = 128 };

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

// The definitions of the following 2 functions appear in `main.cpp`
void clearRows(int *a, size_t m, size_t n, const bool * const foundRows);
void clearColumns(int *a, size_t m, size_t n,
				  const bool * const foundCols,
				  size_t fromCol = 0ULL, size_t toCol = 0ULL);

/**
Here are the ideas behind the adopted solution:

- let the GPU mark columns to reset among a new batch of columns
	while the host performs the actual reset of a previous batch of columns
- after the GPU has finished all column batches, the CPU receives
	the indices of the rows to be reset and performs this task
- the GPU uses 2 streams, to be able to simultaneously process a new column batch
	and copy on host the results for the previous batch
*/
cudaError_t reportAndExpandZeros(int *a, unsigned m, unsigned n,
								 bool *foundRows, bool *foundCols) {
	const unsigned szA = m * n,
		blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;
	int *devA = nullptr;
	bool *devFoundRows = nullptr, *devFoundCols = nullptr;

	cudaStream_t stream[2]; // using 2 streams
	
	// Last of the 2 streams to be used within the loop from below:
	//		0 for odd blocks; 1 for even blocks
	const unsigned lastStream = 1U - blocks & 1U;

	const size_t colsForLastBlock =
		(n % THREADS_PER_BLOCK == 0U) ? THREADS_PER_BLOCK : (n % THREADS_PER_BLOCK);

	CHECK_CUDA_OP(cudaStreamCreate(&stream[0]), LeavingStream0);
	CHECK_CUDA_OP(cudaStreamCreate(&stream[1]), LeavingStream1);

	CHECK_CUDA_OP(cudaMalloc((void**)&devA, szA * sizeof(int)),
				  LeavingA);
	CHECK_CUDA_OP(cudaMalloc((void**)&devFoundRows, m * sizeof(bool)),
				  LeavingFoundRows);
	CHECK_CUDA_OP(cudaMalloc((void**)&devFoundCols, n * sizeof(bool)),
				  LeavingFoundCols);

	// Copy matrix from host on device
	CHECK_CUDA_OP(cudaMemcpy((void*)devA, a, szA * sizeof(int), cudaMemcpyHostToDevice),
				  Leaving);
	
	// Initialize the rows and columns (from the device) to be reported as containing original zeros
	CHECK_CUDA_OP(cudaMemset((void*)devFoundRows, 0, m * sizeof(bool)),
				  Leaving);
	CHECK_CUDA_OP(cudaMemset((void*)devFoundCols, 0, n * sizeof(bool)),
				  Leaving);

	// Alternate streams for handling consecutive blocks of THREADS_PER_BLOCK columns
	for(unsigned blIdx = 0U, strIdx = 0U; blIdx < blocks; ++blIdx, strIdx = 1U - strIdx) {
		// operations using stream[strIdx]

		// After next operation, the found columns from batch blIdx-2 are already on host.
		// Their delivery was planned in the previous turn of this stream
		CHECK_CUDA_OP(cudaStreamSynchronize(stream[strIdx]),
					  Leaving);

		// Launch a single block of THREADS_PER_BLOCK threads in this stream
		markZeros<<<1, THREADS_PER_BLOCK, 0, stream[strIdx]>>>
			(devA, szA, n, blIdx, devFoundRows, devFoundCols);
		CHECK_CUDA_OP(cudaGetLastError(), Leaving);

		// Last block contains the remainder of the columns colsForLastBlock
		const size_t affectedCols =
			((blIdx + 1U == blocks)) ? colsForLastBlock : THREADS_PER_BLOCK;
		CHECK_CUDA_OP(cudaMemcpyAsync((void*)&foundCols[blIdx * THREADS_PER_BLOCK],
			(void*)&devFoundCols[blIdx * THREADS_PER_BLOCK],
			affectedCols * sizeof(bool), cudaMemcpyDeviceToHost, stream[strIdx]),
			Leaving);

		if(blIdx >= 2U) {
			// Expanding zeros on a batch of previously analyzed columns
			// This happens in parallel with the activities from both streams,
			// but after the arrival of the newly found columns on the host
			const size_t fromCol = (blIdx-2U) * THREADS_PER_BLOCK,
				toCol = fromCol + THREADS_PER_BLOCK - 1ULL;
			clearColumns(a, m, n, foundCols, fromCol, toCol);
		}
	}

	// sync with the last stream used, which is 0 for odd blocks and 1 for even blocks
	CHECK_CUDA_OP(cudaStreamSynchronize(stream[lastStream]),
				  Leaving);

	// Now all the remaining marked columns are already copied on host
	// and the marked rows are ready to be copied on host
	
	// Ask for the marked rows, but meanwhile expand the zeros on the remaining found columns
	CHECK_CUDA_OP(cudaMemcpyAsync((void*)foundRows, (void*)devFoundRows, m * sizeof(bool),
		cudaMemcpyDeviceToHost, stream[lastStream]), Leaving);

	clearColumns(a, m, n, foundCols,
				 (blocks > 2U) ? ((blocks-2U) * THREADS_PER_BLOCK) : 0ULL);

	// Make sure the found rows arrived on host and then expand the zeros for them
	CHECK_CUDA_OP(cudaStreamSynchronize(stream[lastStream]),
				  Leaving);
	clearRows(a, m, n, foundRows);

Leaving:
	cudaFree(devFoundCols);
LeavingFoundCols:
	cudaFree(devFoundRows);
LeavingFoundRows:
	cudaFree(devA);
LeavingA:

	cudaStreamDestroy(stream[1]);
LeavingStream1:
	cudaStreamDestroy(stream[0]);
LeavingStream0:

	return cudaGetLastError();
}
