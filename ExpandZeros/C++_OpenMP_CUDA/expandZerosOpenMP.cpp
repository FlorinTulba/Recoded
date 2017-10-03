#include "expandZerosOpenMP.h"
#include "colRanges.h"

#include <omp.h>
#include <cassert>

using namespace std;

void reportAndExpandZerosOpenMP(int *a, long m, long n,
								bool *foundRows, bool *foundCols) {
	assert(nullptr != a && m > 0L && n > 0L && nullptr != foundRows && nullptr != foundCols);

	// Detect first the rows / columns containing the value 0
#pragma omp parallel
#pragma omp for schedule(static, 8) nowait
	for(long r = 0L; r < m; ++r) {
		bool &rowContains0 = foundRows[r];
		long idx = r * n;
		for(long c = 0L; c < n; ++c) {
			if(a[(size_t)idx++] == 0)
				foundCols[(size_t)c] = rowContains0 = true;
		}
	}

	ColRanges colRanges;
	buildColRanges(colRanges, foundCols, (size_t)n);

	const size_t bytesPerRow = sizeof(int) * (size_t)n;

	// Expand the found zeros in a row-wise manner
#pragma omp parallel
#pragma omp for schedule(static, 8) nowait
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
