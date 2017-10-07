/*
Given a matrix, for any element with value 0,
set all elements from the corresponding row and column on 0, as well.

Implementation s using OpenMP and CUDA for NVIDIA GPUs.

@2017 Florin Tulba (florintulba@yahoo.com)
*/

#include "expandZerosCUDA.h"
#include "colRanges.h"

using namespace std;

namespace {
	/// Sets on 0 all the row elements from `foundRows` in matrix `a` [`m` x `n`]
	void clearRows(int *a, size_t m, size_t n, const bool * const foundRows) {
		assert(nullptr != a && nullptr != foundRows);
		const size_t bytesPerRow = n * sizeof(int);
		for(size_t r = 0ULL; r < m; ++r)
			if(foundRows[r]) {
				// Find how many consecutive rows (after r) need to be set on 0
				size_t r1 = r + 1ULL;
				for(; r1 < m && foundRows[r1]; ++r1);
				memset((void*)&a[r * n], 0, bytesPerRow * (r1 - r));
				r = r1;
			}
	}

	/**
	Clears the columns pointed by `foundCols`[`fromCol` : `toCol`] from matrix `a`[`m` x `n`].
	Traverses the matrix in a row-wise fashion to favor obtaining more cache hits.
	*/
	void clearColumns(int *a, size_t m, size_t n,
							 const bool * const foundCols,
							 size_t fromCol = 0ULL, size_t toCol = 0ULL) {
		if(toCol == 0ULL)
			toCol = n - 1ULL;
		assert(nullptr != a && nullptr != foundCols && fromCol <= toCol && toCol < n);

		ColRanges colRanges;
		buildColRanges(colRanges, foundCols, n, fromCol, toCol);
		if(colRanges.empty())
			return;

		for(size_t r = 0ULL, rowStart = 0ULL; r < m; ++r, rowStart += n)
			clearColRangesFromRow(colRanges, &a[rowStart]);
	}
} // anonymous namespace

/**
Here are the ideas behind the adopted solution:

- let the GPU mark columns to reset among a new batch of columns
while the host performs the actual reset of a previous batch of columns
- after the GPU has finished all column batches, the CPU receives
the indices of the rows to be reset and performs this task
- the GPU uses 2 streams, to be able to simultaneously process a new column batch
and copy on host the results for the previous batch
*/
void reportAndExpandZerosCUDA(int *a, unsigned m, unsigned n,
							  bool *foundRows, bool *foundCols) {
	assert(nullptr != a && m > 0U && n > 0U && nullptr != foundRows && nullptr != foundCols);

	const unsigned blocks = (n + THREADS_PER_BLOCK - 1) / THREADS_PER_BLOCK;

	cudaStream_t stream[2]; // using 2 streams

	// Last of the 2 streams to be used within the loop from below:
	//		0 for odd blocks; 1 for even blocks
	const unsigned lastStream = 1U - blocks & 1U;

	const size_t colsForLastBlock =
		(n % THREADS_PER_BLOCK == 0U) ? THREADS_PER_BLOCK : (n % THREADS_PER_BLOCK);

	CudaRAII<cudaStream_t> RAII_Stream0(function<cudaError_t(cudaStream_t*)>(::cudaStreamCreate),
										::cudaStreamDestroy,
										stream[0]);
	CudaRAII<cudaStream_t> RAII_Stream1(function<cudaError_t(cudaStream_t*)>(::cudaStreamCreate),
										::cudaStreamDestroy,
										stream[1]);

	const unsigned szA = m * n;

	// device memory that covers the input matrix and the 2 output vectors
	char *devMem = nullptr; // uses char, as its sizeof is 1, so it's easier to compute the offsets

	// Simulate 3 normal dynamic device memory allocations (aligned to 256B)
	const size_t devMemAreaA = nextMultipleOf<256ULL>(size_t(szA * sizeof(int))),
		devMemFoundCols = nextMultipleOf<256ULL>(size_t(n) * sizeof(bool)),
		totalDevMem = devMemAreaA + devMemFoundCols +
			size_t(m) * sizeof(bool); // device memory for foundRows

	CudaRAII<void*> RAII_devMem(function<cudaError_t(void**, size_t)>(::cudaMalloc),
								::cudaFree,
								(void*&)devMem, totalDevMem);

	int * const devA = reinterpret_cast<int*>(devMem);
	bool * const results = reinterpret_cast<bool*>(&devMem[devMemAreaA]);
	bool * const devFoundCols = results;
	bool * const devFoundRows = &results[devMemFoundCols];

	// Copy matrix from host on device
	CHECK_CUDA_OP(cudaMemcpyAsync((void*)devA, a, szA * sizeof(int), cudaMemcpyHostToDevice, stream[0]));

	// Initialize the rows and columns (from the device) to be reported as containing original zeros
	CHECK_CUDA_OP(cudaMemsetAsync((void*)results, 0, (m + devMemFoundCols) * sizeof(bool), stream[0]));

	static KernelLaunchConfig klc {1U, (unsigned)THREADS_PER_BLOCK, 0U};

	// Alternate streams for handling consecutive blocks of THREADS_PER_BLOCK columns
	for(unsigned blIdx = 0U, strIdx = 0U; blIdx < blocks; ++blIdx, strIdx = 1U - strIdx) {
		// operations using stream[strIdx]

		// After next operation, the found columns from batch blIdx-2 are already on host.
		// Their delivery was planned in the previous turn of this stream
		CHECK_CUDA_OP(cudaStreamSynchronize(stream[strIdx]));

		// Launch a single block of THREADS_PER_BLOCK threads in this stream
		klc.stream = stream[strIdx];
		launchMarkZerosKernel(klc, devA, szA, n, blIdx);
		// the kernel deduces devFoundCols and devFoundRows from devA, szA and n

		// Last block contains the remainder of the columns colsForLastBlock
		const size_t affectedCols =
			((blIdx + 1U == blocks)) ? colsForLastBlock : THREADS_PER_BLOCK;
		CHECK_CUDA_OP(cudaMemcpyAsync((void*)&foundCols[blIdx * THREADS_PER_BLOCK],
			(void*)&devFoundCols[blIdx * THREADS_PER_BLOCK],
			affectedCols * sizeof(bool), cudaMemcpyDeviceToHost, stream[strIdx]));

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
	CHECK_CUDA_OP(cudaStreamSynchronize(stream[lastStream]));

	// Now all the remaining marked columns are already copied on host
	// and the marked rows are ready to be copied on host

	// Ask for the marked rows, but meanwhile expand the zeros on the remaining found columns
	CHECK_CUDA_OP(cudaMemcpyAsync((void*)foundRows, (void*)devFoundRows, m * sizeof(bool),
		cudaMemcpyDeviceToHost, stream[lastStream]));

	clearColumns(a, m, n, foundCols,
				 (blocks > 2U) ? ((blocks-2U) * THREADS_PER_BLOCK) : 0ULL);

	// Make sure the found rows arrived on host and then expand the zeros for them
	CHECK_CUDA_OP(cudaStreamSynchronize(stream[lastStream]));
	clearRows(a, m, n, foundRows);
}
