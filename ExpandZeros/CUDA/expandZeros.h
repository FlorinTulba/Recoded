/*
Given a matrix, for any element with value 0,
set all elements from the corresponding row and column on 0, as well.

Implementation using CUDA for NVIDIA GPUs.

@2017 Florin Tulba (florintulba@yahoo.com)
*/

#ifndef H_EXPAND_ZEROS
#define H_EXPAND_ZEROS

#include <cuda_runtime.h>

/**
Expands the zeros from a[m x n]
Sets on true the indices from foundRows and foundCols where the original zeros were found.
*/
cudaError_t reportAndExpandZeros(int *a, unsigned m, unsigned n,
								 bool *foundRows, bool *foundCols);

#endif // H_EXPAND_ZEROS
