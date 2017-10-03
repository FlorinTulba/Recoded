/*
Given a matrix, for any element with value 0,
set all elements from the corresponding row and column on 0, as well.

Usually the results appear as sparse matrices.
It would make sense:

- (A) either to return just the sequence of the remaining non-zero elements (sparse-matrix format)
- (B) or to copy those non-zero remaining elements into a new matrix (dense-matrix format)

instead of (C) setting on zero probably way more elements from the affected rows / columns.

However, only solution (C) is in-place, which is the reason why it was chosen.

Implementation using CUDA for NVIDIA GPUs.

@2017 Florin Tulba (florintulba@yahoo.com)
*/

#include "expandZerosCUDA.h"
#include "expandZerosOpenMP.h"
#include "timing.h"

#include <vector>
#include <iostream>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <random>
#include <memory>
#include <omp.h>

using namespace std;

/**
Randomly initializes a matrix with elements in 0..1000 range.

@param a the matrix to initialize
@param m the number of rows of the matrix
@param n the number of columns of the matrix
@param dimA the number of elements in `a` [`checkRows`.size() * `checkCols`.size()]
@param origZerosPercentage represents the desired percentage of 0 elements within `a` (range: 0..100)
@param checkRows reports which rows contain elements equal to 0
@param checkCols reports which columns contain elements equal to 0
*/
static void initMat(int *a, size_t m, size_t n, size_t dimA, unsigned origZerosPercentage,
					vector<bool> &checkRows, vector<bool> &checkCols) {
	assert(nullptr != a &&
		   origZerosPercentage <= 100U &&
		   dimA > 0ULL && dimA == m * n);

	static random_device rd;
	static mt19937 randGen(rd());
	static uniform_int_distribution<> uidVal(1, 1000);

	checkRows.clear(); checkRows.resize(m, false);
	checkCols.clear(); checkCols.resize(n, false);

	// Set all elements first on non-zero values
	for(size_t i = 0ULL; i<dimA; ++i)
		a[i] = uidVal(randGen);

	// Choose some coordinates to be set to 0
	uniform_int_distribution<size_t> uidPos(0ULL, dimA-1ULL);
	const size_t zerosCount = size_t(dimA * origZerosPercentage / 100.);
	for(size_t i = 0ULL; i<zerosCount; ++i) {
		const size_t pos = uidPos(randGen),
			row = pos / n, col = pos % n;
		a[pos] = 0;
		checkRows[row] = checkCols[col] = true;
	}
}

/**
Reads a matrix from a stream.
First line needs to contain `m` followed by `n`.
Every following input line must contain the `n` elements from one row of the matrix.

@param pa pointer towards the region of pinned memory allocated for the read matrix
@param m the number of rows of the matrix
@param n the number of columns of the matrix
@param is the source stream of the matrix
*/
static void inputMat(int **pa, size_t &m, size_t &n, istream &is = cin) {
	assert(nullptr != pa);
	string line;
	{ // Reading matrix size from the first line of the file: m followed by n
		if(!getline(is, line))
			throw runtime_error("Couldn't read the first line!");
		istringstream iss(line);
		if((iss>>m>>n).fail())
			throw runtime_error("Couldn't read the size of the matrix!");
		CHECK_CUDA_OP(cudaMallocHost((void**)pa, m * n * sizeof(int))); // pinned memory
	}

	// Reading matrix data, row by row
	for(size_t r = 0ULL, idx = 0ULL; r < m; ++r) {
		if(!getline(is, line))
			throw runtime_error("Couldn't read a new line!");
		istringstream iss(line);
		for(size_t c = 0ULL; c < n; ++c, ++idx) {
			if((iss>>(*pa)[idx]).fail()) // Reading next element from the same row
				throw runtime_error("Couldn't read a matrix element!");
		}
	}
}

/**
Outputs a given matrix to a stream.
First line reports the matrix size: `m` followed by `n`.
Every following input line provides the `n` elements from the corresponding row of the matrix.

@param a the matrix to be presented
@param m the number of rows of the matrix
@param n the number of columns of the matrix
@param os the destination stream for the matrix
*/
static void outputMat(const int * const a, size_t m, size_t n, ostream &os = cout) {
	assert(nullptr != a);
	os<<m<<'\t'<<n<<endl;
	size_t idx = 0ULL;
	for(size_t r = 0ULL; r < m; ++r) {
		for(size_t c = 0ULL; c < n; ++c, ++idx)
			os<<setw(4)<<a[idx]<<" ";
		os<<endl;
	}
	os<<endl;
}

/// @return true if the GPU returns the expected indices for the rows / columns to reset
static bool same(const bool * const found, const vector<bool> &expected) {
	assert(nullptr != found && !expected.empty());
	for(size_t len = expected.size(), i = 0ULL; i < len; ++i)
		if(expected[i] != found[i]) {
			cerr<<"Mismatch: "<<endl<<"\tExpected: ";
			for(size_t j = 0; j < len; ++j)
				cerr<<expected[j];
			cerr<<endl<<"\tReceived: ";
			for(size_t j = 0; j < len; ++j)
				cerr<<found[j];
			cerr<<endl<<endl;
			return false;
		}
	return true;
}

/// Resets the vector of found rows / columns for reprocessing by a different algorithm
static void clearFound(bool * const found, size_t count) {
	for(size_t i = 0ULL; i < count; ++i)
		found[i] = false;
}

/**
Defines a matrix A[m rows by n columns] and sends it to the GPU.
Then it performs the following loop:
- waits for the GPU to detect several new
*/
void main() {
	try {
		// Ensure control over the number of threads during the application
		if(omp_get_dynamic())
			omp_set_dynamic(0);

		// Use as many threads as processors
		omp_set_num_threads(omp_get_num_procs());

		// Ensure no nested parallelism
		if(omp_get_nested())
			omp_set_nested(0);

		CudaSession cudaSession;

		enum { TIMES = 1000 }; // iterations count

		// Max matrix dimensions
		enum { mMax = 500, nMax = 500, dimAMax = mMax*nMax };

		enum { origZerosPercentage = 4 }; // Desired percentage of zeros within generated matrices

		random_device rd;
		mt19937 randGen(rd());
		uniform_int_distribution<size_t>
			mRand(15, mMax-1), nRand(15, nMax-1);

		const unique_ptr<int[]> origMat = make_unique<int[]>(size_t(dimAMax * sizeof(int)));
		int *a = nullptr;
		CudaRAII<void*> pinnedMatA(function<cudaError_t(void**, size_t)>(::cudaMallocHost),
								   ::cudaFreeHost,
								   (void*&)a, size_t(dimAMax * sizeof(int)));

		bool *foundRows = nullptr, *foundCols = nullptr;
		CudaRAII<void*> pinnedFoundRows(function<cudaError_t(void**, size_t)>(::cudaMallocHost),
										::cudaFreeHost,
										(void*&)foundRows, mMax * sizeof(bool));
		CudaRAII<void*> pinnedFoundCols(function<cudaError_t(void**, size_t)>(::cudaMallocHost),
										::cudaFreeHost,
										(void*&)foundCols, nMax * sizeof(bool));

		// Testing TIMES matrices of random sizes and with random elements
		Timer timerCUDA(false), timerOpenMP(false);
		size_t totalElems = 0ULL; // Count of the analyzed elements of all matrices from all iterations
		vector<bool> checkRows, checkCols; // Correct rows / columns containing values of 0
		for(int i = 0; i < TIMES; ++i) {
			// Init random dimensions and matrix
			const size_t m = mRand(randGen),
				n = nRand(randGen),
				dimMat = m * n;
			totalElems += dimMat;
			initMat(origMat.get(), m, n, dimMat, origZerosPercentage, checkRows, checkCols);

			// copy the original to be modified by the CUDA algorithm version
			memcpy_s((void*)a, size_t(dimAMax * sizeof(int)),
					 (const void*)origMat.get(), size_t(dimMat * sizeof(int)));

			// Expand the zeros from the matrix while timing the operation
			timerCUDA.resume();
			reportAndExpandZerosCUDA(a, (unsigned)m, (unsigned)n, foundRows, foundCols);
			timerCUDA.pause();

			// Validate result
			assert(same(foundRows, checkRows));
			assert(same(foundCols, checkCols));

			// Test using OpenMP
			clearFound(foundRows, m);
			clearFound(foundCols, n);

			// Expand the zeros directly from the original matrix while timing the operation
			timerOpenMP.resume();
			reportAndExpandZerosOpenMP(origMat.get(), (long)m, (long)n, foundRows, foundCols);
			timerOpenMP.pause();

			// Validate result
			assert(same(foundRows, checkRows));
			assert(same(foundCols, checkCols));
		}

		cout<<"Total time CUDA: "<<timerCUDA.elapsed()<<"s for "<<totalElems<<" elements in "
			<<TIMES<<" matrices, which means "<<timerCUDA.elapsed() * 1e9 / totalElems<<"ns/element"<<endl;
		cout<<"Total time OpenMP: "<<timerOpenMP.elapsed()<<"s for "<<totalElems<<" elements in "
			<<TIMES<<" matrices, which means "<<timerOpenMP.elapsed() * 1e9 / totalElems<<"ns/element"<<endl;

		timerCUDA.done();
		timerOpenMP.done();
	} catch(runtime_error &e) {
		cerr<<e.what()<<endl;
	}
}
