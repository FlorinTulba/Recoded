/*
Given a matrix, for any element with value 0,
set all elements from the corresponding row and column on 0, as well.

Implementations using OpenMP and CUDA for NVIDIA GPUs.

@2017 Florin Tulba (florintulba@yahoo.com)
*/

#include "expandZerosOpenMP.h"
#include "colRanges.h"
#include "align.h"
#include "../../common/config.h"

#include <omp.h>
#include <algorithm>
#include <cassert>

using namespace std;

namespace {
	/// Initialization and details for OpenMP
	struct OpenMP_session {
		/// Provides the number of CPU-s from the system
		const int processorsCount = omp_get_num_procs();

		/// Ensures no nested parallelism
		OpenMP_session() {
			if(omp_get_nested())
				omp_set_nested(0);
		}
	};
	const OpenMP_session ompSession;

	// Don't initialize the singleton and read the value within reportAndExpandZerosOpenMP,
	// as this shouldn't be timed
	const size_t minElemsPerThread =
		Config::get().valueOf(ConfigItem<size_t>("MinElemsPerOpenMPThread"), 0ULL);
} // anonymous namespace

void reportAndExpandZerosOpenMP(int *a, long m, long n,
								bool *foundRows, bool *foundCols) {
	assert(nullptr != a && m > 0L && n > 0L && nullptr != foundRows && nullptr != foundCols);

	// There is no point spawning a thread unless it has at least minElemsPerThread elements to analyze.
	// The main thread should process most of the time fewer elements,
	// as it also needs to wait for the created threads
	const int rowsPerThread =
		(int)min((double)m,
				ceil(max((double)m / ompSession.processorsCount,
						minElemsPerThread / (double)n)));
	const int requiredThreads = (int)ceil(m / (double)rowsPerThread);
	if(requiredThreads > 1) { // multi-threaded region using `localFoundCols`	
#pragma omp parallel num_threads(requiredThreads)
		{
			// Avoids false sharing of foundCols through a local vector for each thread
			vector<bool> localFoundCols((size_t)n, false); // similar technique with reduction

			// The scheduled chunk size from the loop below seems less important.
			// However, a value of l1CacheLineSz() / sizeof(bool)
			// would prevent completely the false sharing of foundRows,
			// if foundRows is l1CacheLineSz() - aligned.

			// Detect first the rows / columns containing the value 0
#pragma omp for schedule(static, rowsPerThread) nowait
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

	} else { // when sequential, `localFoundCols` from the parallel region (from above) isn't necessary
		for(long r = 0L; r < m; ++r) {
			bool rowContains0 = false;
			long idx = r * n;
			for(long c = 0L; c < n; ++c) {
				if(a[(size_t)idx++] == 0)
					foundCols[(size_t)c] = rowContains0 = true;
			}
			if(rowContains0)
				foundRows[r] = true; // setting this outside the inner loop minimizes false sharing of foundRows
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
#pragma omp parallel num_threads(requiredThreads)
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
