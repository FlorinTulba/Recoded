/*
Given a matrix, for any element with value 0,
set all elements from the corresponding row and column on 0, as well.

Implementation using CUDA for NVIDIA GPUs.

@2017 Florin Tulba (florintulba@yahoo.com)
*/

#include "cudaSession.h"
#include "markZeros.h"
#include "timing.h"

#include <vector>
#include <iostream>
#include <iomanip>
#include <algorithm>
#include <random>
#include <cassert>

using namespace std;

void initMat(int *a, size_t dimA, unsigned origZerosPercentage,
			 vector<bool> &checkRows, vector<bool> &checkCols) {
	random_device rd;
	mt19937 randGen(rd());
	uniform_int_distribution<> uidVal(1, 1000);
	for(size_t i = 0ULL; i<dimA; ++i)
		a[i] = uidVal(randGen);

	const size_t n = checkCols.size();
	uniform_int_distribution<size_t> uidPos(0ULL, dimA-1ULL);
	size_t zerosCount = size_t(dimA * origZerosPercentage / 100.);
	for(size_t i = 0ULL; i<zerosCount; ++i) {
		const size_t pos = uidPos(randGen),
			row = pos / n, col = pos % n;
		a[pos] = 0;
		checkRows[row] = checkCols[col] = true;
	}
}

void showMat(const int * const a, size_t m, size_t n) {
	size_t idx = 0ULL;
	for(size_t r = 0ULL; r < m; ++r) {
		for(size_t c = 0ULL; c < n; ++c, ++idx)
			cout<<setw(4)<<a[idx]<<" ";
		cout<<endl;
	}
	cout<<endl;
}

void clearRows(int *a, size_t m, size_t n, const bool * const foundRows) {
	const size_t bytesPerRow = n * sizeof(int);
	for(size_t r = 0ULL; r < m; ++r)
		if(foundRows[r])
			memset((void*)&a[r * n], 0, bytesPerRow);
}

void clearColumns(int *a, size_t m, size_t n, const bool * const foundCols) {
	vector<size_t> colIndices;
	colIndices.reserve(n);
	for(size_t c = 0ULL; c < n; ++c)
		if(foundCols[c])
			colIndices.push_back(c);
	const size_t totalCols = colIndices.size();
	for(size_t r = 0ULL, rowStart = 0ULL; r < m; ++r, rowStart += n)
		for(size_t cIdx = 0ULL; cIdx < totalCols; ++cIdx)
			a[rowStart + colIndices[cIdx]] = 0;
}

bool same(const bool * const found, const vector<bool> &expected) {
	for(size_t len = expected.size(), i = 0ULL; i < len; ++i)
		if(expected[i] != found[i]) {
/*
			cerr<<"Mismatch: "<<endl<<"\tExpected: ";
			for(size_t j = 0; j < len; ++j)
				cerr<<expected[j];
			cerr<<endl<<"\tReceived: ";
			for(size_t j = 0; j < len; ++j)
				cerr<<found[j];
			cerr<<endl<<endl;
*/
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

	enum { TIMES = 10000 };
	enum { mMax = 500, nMax = 500, dimAMax = mMax*nMax };
	enum { origZerosPercentage = 4 };
	int *a = new int[dimAMax];
	bool *foundRows = new bool[mMax],
		*foundCols = new bool[nMax];
	vector<bool> checkRows, checkCols;

	random_device rd;
	mt19937 randGen(rd());
	uniform_int_distribution<size_t>
		mRand(15, mMax-1), nRand(15, nMax-1);

	Timer timer(false);
	size_t totalElems = 0ULL;
	for(int i = 0; i < TIMES; ++i) {
		const size_t m = mRand(randGen), n = nRand(randGen),
			dimA = m * n;
		totalElems += dimA;
		checkRows.clear(); checkRows.resize(m);
		checkCols.clear(); checkCols.resize(n);
		initMat(a, dimA, origZerosPercentage, checkRows, checkCols);
		//showMat(a, m, n);

		timer.resume();
		findZeros(a, (unsigned)m, (unsigned)n, foundRows, foundCols);
		timer.pause();

		assert(same(foundRows, checkRows));
		assert(same(foundCols, checkCols));

		clearRows(a, m, n, foundRows);
		clearColumns(a, m, n, foundCols);
		//showMat(a, m, n);
	}

	timer.done();
	cout<<"Total time: "<<timer.elapsed()<<"s for "<<totalElems<<" elements in "
		<<TIMES<<" matrices, which means "<<timer.elapsed() * 1e9 / totalElems<<"ns/element"<<endl;

	delete[] foundCols;
	delete[] foundRows;
	delete[] a;
}
