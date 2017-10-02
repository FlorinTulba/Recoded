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

#include "cudaSession.h"
#include "expandZeros.h"
#include "timing.h"

#include <vector>
#include <iostream>
#include <sstream>
#include <fstream>
#include <iomanip>
#include <algorithm>
#include <random>
#include <cassert>

using namespace std;

/**
Randomly initializes a matrix with elements in 0..1000 range.

@param a the matrix to initialize
@param dimA the number of elements in `a` [`checkRows`.size() * `checkCols`.size()]
@param origZerosPercentage represents the desired percentage of 0 elements within `a` (range: 0..100)
@param checkRows reports which rows contain elements equal to 0
@param checkCols reports which columns contain elements equal to 0
*/
void initMat(int *a, size_t dimA, unsigned origZerosPercentage,
			 vector<bool> &checkRows, vector<bool> &checkCols) {
	assert(nullptr != a &&
		   origZerosPercentage <= 100U &&
		   dimA > 0ULL && dimA == checkRows.size() * checkCols.size());

	static random_device rd;
	static mt19937 randGen(rd());
	static uniform_int_distribution<> uidVal(1, 1000);

	const size_t n = checkCols.size();

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
void inputMat(int **pa, size_t &m, size_t &n, istream &is = cin) {
	assert(nullptr != pa);
	string line;
	{ // Reading matrix size from the first line of the file: m followed by n
		if(!getline(is, line)) {
			cerr<<"Couldn't read the first line!"<<endl;
			throw runtime_error("Couldn't read the first line!");
		}
		istringstream iss(line);
		if((iss>>m>>n).fail()) {
			cerr<<"Couldn't read the size of the matrix!"<<endl;
			throw runtime_error("Couldn't read the size of the matrix!");
		}
		CHECK_CUDA_OP(cudaMallocHost((void**)pa, m * n * sizeof(int)), // pinned memory
					  AllocError); // jumps to AllocError on allocation error
		goto AllocOk;

	AllocError:
		cerr<<"Couldn't allocate "<<m*n<<"B of pinned memory for the matrix!"<<endl;
		throw runtime_error("Couldn't allocate pinned memory for the matrix!");

	AllocOk:
		;
	}

	// Reading matrix data, row by row
	for(size_t r = 0ULL, idx = 0ULL; r < m; ++r) {
		if(!getline(is, line)) {
			cerr<<"Couldn't read a new line!"<<endl;
			throw runtime_error("Couldn't read a new line!");
		}
		istringstream iss(line);
		for(size_t c = 0ULL; c < n; ++c, ++idx) {
			if((iss>>(*pa)[idx]).fail()) { // Reading next element from the same row
				cerr<<"Couldn't read element from row "<<r<<" and column "<<c<<endl;
				throw runtime_error("Couldn't read a matrix element!");
			}
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
void outputMat(const int * const a, size_t m, size_t n, ostream &os = cout) {
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

	// Mapping: start column - count of consecutive columns to reset
	vector<pair<size_t, size_t>> colRanges;
	colRanges.reserve(toCol - fromCol + 1ULL);
	for(size_t c = fromCol; c <= toCol; ++c)
		if(foundCols[c]) {
			// Find how many consecutive columns (after c) need to be set on 0
			size_t c1 = c + 1ULL;
			for(; c1 <= toCol && foundCols[c1]; ++c1);
			colRanges.emplace_back(c, c1 - c);
			c = c1;
		}
	if(colRanges.empty())
		return;
	for(size_t r = 0ULL, rowStart = 0ULL; r < m; ++r, rowStart += n)
		for(const auto &colRange : colRanges)
			memset((void*)&a[rowStart + colRange.first], 0,
					sizeof(int) * colRange.second); // sets on zero a range of consecutive columns
}

/// @return true if the GPU returns the expected indices for the rows / columns to reset
bool same(const bool * const found, const vector<bool> &expected) {
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

/**
Defines a matrix A[m rows by n columns] and sends it to the GPU.
Then it performs the following loop:
- waits for the GPU to detect several new
*/
void main() {
	CudaSession cudaSession;

	enum { TIMES = 1000 }; // iterations count

	// Max matrix dimensions
	enum { mMax = 500, nMax = 500, dimAMax = mMax*nMax };
	
	enum { origZerosPercentage = 4 }; // Desired percentage of zeros within generated matrices

	vector<bool> checkRows, checkCols;

	random_device rd;
	mt19937 randGen(rd());
	uniform_int_distribution<size_t>
		mRand(15, mMax-1), nRand(15, nMax-1);

	Timer timer(false);
	size_t totalElems = 0ULL; // Count of the analyzed elements of all matrices from all iterations

// 	int *a = new int[dimAMax]; // non-pinned
	int *a = nullptr;
	CHECK_CUDA_OP(cudaMallocHost((void**)&a, dimAMax * sizeof(int)), // pinned memory
				  Return);
	
// 	bool *foundRows = new bool[mMax], *foundCols = new bool[nMax]; // non-pinned
	bool *foundRows = nullptr, *foundCols = nullptr;
	CHECK_CUDA_OP(cudaMallocHost((void**)&foundRows, mMax * sizeof(bool)), // pinned memory
				  FreeA);
	CHECK_CUDA_OP(cudaMallocHost((void**)&foundCols, nMax * sizeof(bool)), // pinned memory
				  FreeA_Rows);

	// Testing TIMES matrices of random sizes and with random elements
	for(int i = 0; i < TIMES; ++i) {
		// Init random dimensions and matrix
		const size_t m = mRand(randGen),
			n = nRand(randGen),
			dimA = m * n;
		totalElems += dimA;
		checkRows.clear(); checkRows.resize(m, false);
		checkCols.clear(); checkCols.resize(n, false);
		initMat(a, dimA, origZerosPercentage, checkRows, checkCols);

		// Expand the zeros from the matrix while timing the operation
		timer.resume();
		reportAndExpandZeros(a, (unsigned)m, (unsigned)n, foundRows, foundCols);
		timer.pause();

		// Validate result
		assert(same(foundRows, checkRows));
		assert(same(foundCols, checkCols));
	}

	cout<<"Total time: "<<timer.elapsed()<<"s for "<<totalElems<<" elements in "
		<<TIMES<<" matrices, which means "<<timer.elapsed() * 1e9 / totalElems<<"ns/element"<<endl;

	timer.done();

// 	delete[] foundCols; // non-pinned
	cudaFreeHost((void*)foundCols);

FreeA_Rows:
// 	delete[] foundRows; // non-pinned
	cudaFreeHost((void*)foundRows);

FreeA:
// 	delete[] a; // non-pinned
	cudaFreeHost((void*)a);

Return:
	;
}
