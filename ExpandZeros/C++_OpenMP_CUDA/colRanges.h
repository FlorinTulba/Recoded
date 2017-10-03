#ifndef H_COL_RANGES
#define H_COL_RANGES

#include <vector>

/// Mapping: start column - count of consecutive columns to reset
typedef std::vector<std::pair<size_t, size_t>> ColRanges;

/**
Traverses `foundCols`[`fromCol` : `toCol`] and merges all consecutive columns containing true.
So it creates a set of ranges as explained for type `ColRanges`
*/
void buildColRanges(ColRanges &colRanges, const bool * const foundCols, size_t n,
					size_t fromCol = 0ULL, size_t toCol = 0ULL);

/// Clears the elements from a row for the columns that contained value 0
void clearColRangesFromRow(const ColRanges &colRanges, int * const rowStart);

#endif // H_COL_RANGES
