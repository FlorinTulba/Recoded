#ifndef H_EXPAND_ZEROS_OPENMP
#define H_EXPAND_ZEROS_OPENMP

/**
Expands the zeros from a[m x n]
Sets on true the indices from foundRows and foundCols where the original zeros were found.
*/
void reportAndExpandZerosOpenMP(int *a, long m, long n,
								bool *foundRows, bool *foundCols);

#endif // H_EXPAND_ZEROS_OPENMP
