/*
Given a matrix, for any element with value 0,
set all elements from the corresponding row and column on 0, as well.

Implementations using OpenMP and CUDA for NVIDIA GPUs.

@2017 Florin Tulba (florintulba@yahoo.com)
*/

#ifndef H_EXPAND_ZEROS_OPENMP
#define H_EXPAND_ZEROS_OPENMP

/**
Expands the zeros from a[m x n]
Sets on true the indices from foundRows and foundCols where the original zeros were found.
*/
void reportAndExpandZerosOpenMP(int *a, long m, long n,
								bool *foundRows, bool *foundCols);

#endif // H_EXPAND_ZEROS_OPENMP
