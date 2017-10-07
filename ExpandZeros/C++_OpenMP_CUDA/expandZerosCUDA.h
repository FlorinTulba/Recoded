/*
Given a matrix, for any element with value 0,
set all elements from the corresponding row and column on 0, as well.

Implementations using OpenMP and CUDA for NVIDIA GPUs.

@2017 Florin Tulba (florintulba@yahoo.com)
*/

#ifndef H_EXPAND_ZEROS_CUDA
#define H_EXPAND_ZEROS_CUDA

#include "cudaSession.h"

enum {
	THREADS_PER_BLOCK = 384 ///< how many threads does a block contain
};

/**
Expands the zeros from a[m x n]
Sets on true the indices from foundRows and foundCols where the original zeros were found.
*/
void reportAndExpandZerosCUDA(int *a, unsigned m, unsigned n,
							  bool *foundRows, bool *foundCols);

/**
Launches markZeros CUDA kernel

@param kernelLaunchConfig the launch configuration for the kernel
@param a global memory containing the matrix to process.
After the bytes for a, there are 2 regions for foundCols and foundRows.
@param szA the number of elements of `a`
@param n the number of columns of `a`
@param blIdx the index of a batch of consecutive columns to be analyzed within a separate stream
by a block of THREADS_PER_BLOCK threads
*/
void launchMarkZerosKernel(const KernelLaunchConfig &kernelLaunchConfig,
						   const int * const a,
						   unsigned szA, unsigned n, unsigned blIdx);

#endif // H_EXPAND_ZEROS_CUDA
