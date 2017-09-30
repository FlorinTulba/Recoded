/*
Given a matrix, for any element with value 0,
set all elements from the corresponding row and column on 0, as well.

Implementation using CUDA for NVIDIA GPUs.

@2017 Florin Tulba (florintulba@yahoo.com)
*/

#ifndef H_CUDA_SESSION
#define H_CUDA_SESSION

/// Sets up a CUDA session
class CudaSession {
public:
	CudaSession(); ///< sets the current device

	/// cudaDeviceReset must be called before exiting in order for profiling and
	/// tracing tools such as Nsight and Visual Profiler to show complete traces.
	~CudaSession();
};

#endif // H_CUDA_SESSION
