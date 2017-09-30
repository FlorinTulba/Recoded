/*
Given a matrix, for any element with value 0,
set all elements from the corresponding row and column on 0, as well.

Implementation using CUDA for NVIDIA GPUs.

@2017 Florin Tulba (florintulba@yahoo.com)
*/

#ifndef H_MARK_ZEROS
#define H_MARK_ZEROS

#include <cuda_runtime.h>

/**
Marks zeros from a[m x n]
in foundRows and foundCols.
*/
cudaError_t findZeros(const int *a, unsigned m, unsigned n,
					  bool *foundRows, bool *foundCols);

#endif // H_MARK_ZEROS
