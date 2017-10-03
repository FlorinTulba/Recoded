#include "colRanges.h"

#include <cassert>

using namespace std;

void buildColRanges(ColRanges &colRanges, const bool * const foundCols, size_t n,
					size_t fromCol/* = 0ULL*/, size_t toCol/* = 0ULL*/) {
	if(toCol == 0ULL)
		toCol = n - 1ULL;
	assert(foundCols != nullptr && fromCol <= toCol && toCol < n && colRanges.empty());

	colRanges.reserve(toCol - fromCol + 1ULL);
	for(size_t c = fromCol; c <= toCol; ++c)
		if(foundCols[c]) {
			// Find how many consecutive columns (after c) need to be set on 0
			size_t c1 = c + 1ULL;
			for(; c1 <= toCol && foundCols[c1]; ++c1);
			colRanges.emplace_back(c, c1 - c);
			c = c1;
		}
}

void clearColRangesFromRow(const ColRanges &colRanges, int * const rowStart) {
	assert(!colRanges.empty()); // check this before calling the method for each row
	assert(nullptr != rowStart);

	for(const auto &colRange : colRanges)
		memset((void*)&rowStart[colRange.first], 0,
				sizeof(int) * colRange.second); // sets on zero a range of consecutive columns
}