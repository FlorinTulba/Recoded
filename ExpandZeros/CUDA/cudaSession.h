/*
Given a matrix, for any element with value 0,
set all elements from the corresponding row and column on 0, as well.

Implementation using CUDA for NVIDIA GPUs.

@2017 Florin Tulba (florintulba@yahoo.com)
*/

#ifndef H_CUDA_SESSION
#define H_CUDA_SESSION

#include <functional>
#include <sstream>
#include <cassert>

#include <cuda_runtime.h>

// Macro for checking the execution of a CUDA instruction
// When not successful, it throws runtime_error providing an explanation
#define CHECK_CUDA_OP(op) \
	if(op, cudaGetLastError() != cudaSuccess) { \
		std::ostringstream oss; \
		oss<<"Operation " #op " failed because: " \
			<<cudaGetErrorString(cudaGetLastError()); \
		throw runtime_error(oss.str()); \
	}

// Macro for checking if the launch of a CUDA kernel succeeded.
// When not successful, it throws runtime_error providing an explanation
#define CHECK_CUDA_KERNEL_LAUNCH(kernelName) \
	if(cudaGetLastError() != cudaSuccess) { \
		std::ostringstream oss; \
		oss<<"The launch of kernel " #kernelName " failed because: " \
			<<cudaGetErrorString(cudaGetLastError()); \
		throw runtime_error(oss.str()); \
	}

/**
RAII for CUDA resources

Ensures releasing / stack unwinding for the CUDA resources allocated in this manner.
*/
template<typename T>
class CudaRAII {
protected:
	T &resource; ///< the protected resource to be released
	const std::function<cudaError_t(T)> destructFn; ///< the releasing function

public:
	/// Tries to allocate the resource based on the provided `args`
	template<class ... ArgTypes>
	CudaRAII(const std::function<cudaError_t(T*, ArgTypes ...)> &constructFn, ///< the function for allocating the resource
			 const std::function<cudaError_t(T)> &destructFn_, ///< the function for releasing the resource
			 T &resource_,	///< reference to the resource to be allocated / released
			 ArgTypes ... args ///< the rest of the arguments required by `constructFn`
			 ) : resource(resource_), destructFn(destructFn_) {

		if(constructFn(&resource, std::forward<ArgTypes>(args)...) != cudaSuccess) {
			std::ostringstream oss;
			oss<<cudaGetErrorString(cudaGetLastError())<<" while performing " __FUNCTION__;
			throw std::runtime_error(oss.str());
		}
	}

	/// Releases the resource
	~CudaRAII() {
		destructFn(resource);
	}
};

/// Kernel launch configuration
struct KernelLaunchConfig {
	unsigned blocksCount;		///< the number of blocks to execute the kernel
	unsigned threadsPerBlock;	///< how many threads does a block contain
	unsigned shMemSz;			///< the size of the shared memory for a block
	cudaStream_t stream;		///< the stream where to execute the kernel

	KernelLaunchConfig(unsigned blocksCount_ = 1U,
					   unsigned threadsPerBlock_ = 32U,
					   unsigned shMemSz_ = 0U,
					   cudaStream_t stream_ = nullptr) :
			blocksCount(blocksCount_),
			threadsPerBlock(threadsPerBlock_),
			shMemSz(shMemSz_),
			stream(stream_) {
		assert(blocksCount > 0U && threadsPerBlock > 0U);
	}
};

/// Sets up a CUDA session
class CudaSession {
public:
	CudaSession(); ///< sets the current device

	/// cudaDeviceReset must be called before exiting in order for profiling and
	/// tracing tools such as Nsight and Visual Profiler to show complete traces.
	~CudaSession();
};

#endif // H_CUDA_SESSION
