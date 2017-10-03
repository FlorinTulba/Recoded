/*
Given a matrix, for any element with value 0,
set all elements from the corresponding row and column on 0, as well.

Implementations using OpenMP and CUDA for NVIDIA GPUs.

@2017 Florin Tulba (florintulba@yahoo.com)
*/

#include "expandZerosOpenMP.h"
#include "colRanges.h"
#include "align.h"

#include <omp.h>
#include <cassert>

using namespace std;

void reportAndExpandZerosOpenMP(int *a, long m, long n,
								bool *foundRows, bool *foundCols) {
	assert(nullptr != a && m > 0L && n > 0L && nullptr != foundRows && nullptr != foundCols);

#pragma omp parallel
	{
		// Avoids false sharing of foundCols through a local vector for each thread
		vector<bool> localFoundCols((size_t)n, false); // similar technique with reduction

		// The scheduled chunk size from the loop below seems less important.
		// However, a value of l1CacheLineSz() / sizeof(bool)
		// would prevent completely the false sharing of foundRows,
		// if foundRows is l1CacheLineSz() - aligned.

		// Detect first the rows / columns containing the value 0
#pragma omp for schedule(static, 8) nowait
		for(long r = 0L; r < m; ++r) {
			bool rowContains0 = false;
			long idx = r * n;
			for(long c = 0L; c < n; ++c) {
				if(a[(size_t)idx++] == 0)
					localFoundCols[(size_t)c] = rowContains0 = true;
			}
			if(rowContains0)
				foundRows[r] = true; // setting this outside the inner loop minimizes false sharing of foundRows
		}

		// Perform an union of all localFoundCols into foundCols
		for(long c = 0L; c < n; ++c) {
			if(localFoundCols[c] && !foundCols[c]) {
				// Minimizes false sharing of foundCols by reducing writing operations.
				// Overwriting events actually don't change the data,
				// but they invalidate the corresponding L1 cache line from other threads.
				// This is however cheaper than a synchronization mechanism.
				foundCols[c] = true;
//#pragma omp flush(foundCols[c]) // cannot flush the element of an array
			}
		}
	}

	ColRanges colRanges;
	buildColRanges(colRanges, foundCols, (size_t)n);

	const size_t bytesPerRow = sizeof(int) * (size_t)n;

	// The scheduled chunk size ensures the horizontal chunks fall always in distinct L1 cache lines:
	// - a is allocated as aligned to l1CacheLineSz() and contains integers
	// - every row contains sizeof(int) * n bytes
	// - a chunk contains l1CacheLineSz() / sizeof(int) rows
	// So, a chunk contains n * l1CacheLineSz() bytes.
	// This guarantees there is no false sharing.
	static const size_t chunkSize = l1CacheLineSz() / sizeof(int);

	// Expand the found zeros in a row-wise manner
#pragma omp parallel
#pragma omp for schedule(static, chunkSize) nowait
	for(long r = 0L; r < m; ++r) {
		int * const rowStart = &a[size_t(r * n)];

		// Not using the merge of consecutive rows containing value 0 (technique used by the CUDA algorithm)
		// since the merge might cover rows tackled by a different thread
		if(foundRows[(size_t)r])
			memset((void*)rowStart, 0, bytesPerRow);
		else
			clearColRangesFromRow(colRanges, rowStart);
	}
}
