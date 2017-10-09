/*
Given a matrix, for any element with value 0,
set all elements from the corresponding row and column on 0, as well.

Implementation s using OpenMP and CUDA for NVIDIA GPUs.

@2017 Florin Tulba (florintulba@yahoo.com)
*/

#include "expandZerosCUDA.h"
#include "colRanges.h"
#include "../../common/config.h"

using namespace std;

namespace {
	// Don't initialize the singleton and read the value within reportAndExpandZerosCUDA,
	// as this shouldn't be timed
	const size_t minElemsForGPU = Config::get().valueOf(ConfigItem<size_t>("MinElemsForGPU"), 0ULL);
	const double workQuotaGPU = Config::get().valueOf(ConfigItem<double>("WorkQuotaGPU"), 0.);

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

	// Initial kernel launch configuration
	KernelLaunchConfig klc { 1U, (unsigned)THREADS_PER_BLOCK, 0U, nullptr };
} // anonymous namespace

extern void reportAndExpandZerosOpenMP(int *a, long m, long n,
									   bool *foundRows, bool *foundCols);

/**
Here are the ideas behind the adopted solution:

- let the GPU mark rows and columns containing value 0 from the top of the matrix
- let the CPU process the rest of the matrix
- when the GPU reports the found rows and columns in the top part of the matrix,
	the CPU needs to update any additionally detected row / column
*/
void reportAndExpandZerosCUDA(const CudaSession &cudaSession,
							  int *a, unsigned m, unsigned n,
							  bool *resultsHost) {
	assert(nullptr != a && m > 0U && n > 0U && nullptr != resultsHost);

	// Compute how many rows to be computed on GPU
	const size_t mDev = (size_t)ceil(m * workQuotaGPU);
	if(mDev * n < minElemsForGPU) { // Don't use the GPU for only a few elements. Use OpenMP instead
		reportAndExpandZerosOpenMP(a, (long)m, (long)n, &resultsHost[(size_t)n], resultsHost);
		return;
	}

	// device memory that covers the assigned part of the input matrix and the 2 output vectors
	// uses char, as its sizeof is 1, so it's easier to compute the offsets
	char* devMem = cudaSession.getReservedMem();
	assert(devMem != nullptr);

	const unsigned elemsDevA = (unsigned)mDev * n; // the elements assigned to the GPU
	const size_t szDevA = size_t(elemsDevA) * sizeof(int);

	// Map the reserved device memory for the assigned part from the input and for foundCols and foundRows
	const size_t devMemAreaA = nextMultipleOf<256ULL>(szDevA); // padding the area for matrix a
	int * const devA = (int*)devMem;
	bool * const results = (bool*)&devMem[devMemAreaA]; // the results come after padding the area for matrix a
	bool * const devFoundRows = &results[size_t(n)]; // foundCols and foundRows are consecutive areas

	// Copy the assigned part from the input matrix from host to device
	CHECK_CUDA_OP(cudaMemcpyAsync((void*)devA, a, szDevA, cudaMemcpyHostToDevice));

	// Initialize the rows (from the device) to be reported as containing original zeros.
	// Columns don't need initialization, as each element is appropriately set by the kernel
	CHECK_CUDA_OP(cudaMemsetAsync((void*)devFoundRows, 0, mDev * sizeof(bool)));

	klc.setBlocksCount((n - 1U) / (unsigned)THREADS_PER_BLOCK + 1U);
	launchMarkZerosKernel(klc, devA, elemsDevA, n);
	// the kernel deduces devFoundCols and devFoundRows from devA, elemsDevA and n

	// Ask for the marked rows, but meanwhile expand the zeros on the remaining found columns
	CHECK_CUDA_OP(cudaMemcpyAsync((void*)resultsHost, (void*)results, (mDev + n) * sizeof(bool),
		cudaMemcpyDeviceToHost));

	// Process the remaining part of the input matrix on CPU

	// CPU returns the found columns in a separate array
	const unique_ptr<bool[]> foundColsCpu = make_unique<bool[]>((size_t)n);
	memset((void*)foundColsCpu.get(), 0, (size_t)n);
	bool * const foundRowsCpu = &resultsHost[(size_t)n + mDev];
	memset((void*)foundRowsCpu, 0, (size_t)m - mDev);
	reportAndExpandZerosOpenMP(&a[(size_t)elemsDevA], (long)m - (long)mDev, (long)n,
							   foundRowsCpu, foundColsCpu.get());

	CHECK_CUDA_OP(cudaStreamSynchronize(nullptr));
	bool * const foundCols = resultsHost,
		*const foundRows = &resultsHost[(size_t)n];
	const size_t bytesPerRow = sizeof(int) * (size_t)n;

	// Apply the findings reported by the GPU:
	// - merge the columns and update the upper part of the matrix (both rows and columns)
	// - update only the extra columns spotted by GPU from the lower part of matrix
	vector<unsigned> extraColumns;
	extraColumns.reserve((size_t)n);
	for(size_t c = 0ULL; c < (size_t)n; ++c) {
		if(!foundCols[c]) {
			if(foundColsCpu[c])
				foundCols[c] = true;

		} else if(!foundColsCpu[c])
			extraColumns.push_back((unsigned)c);
	}

	ColRanges colRanges;
	buildColRanges(colRanges, foundCols, (size_t)n);

	size_t r = 0ULL;
	for(; r < mDev; ++r) {
		int * const rowStart = &a[size_t(r * n)];

		// Not using the merge of consecutive rows containing value 0 (technique used by the CUDA algorithm)
		// since the merge might cover rows tackled by a different thread
		if(foundRows[(size_t)r])
			memset((void*)rowStart, 0, bytesPerRow);
		else
			clearColRangesFromRow(colRanges, rowStart);
	}

	if(!extraColumns.empty())
		return;
	assert(r == mDev);
	for(; r < m; ++r) {
		if(foundRows[(size_t)r])
			continue;

		int * const rowStart = &a[size_t(r * n)];
		for(unsigned c : extraColumns)
			rowStart[(size_t)c] = 0;
	}
}
