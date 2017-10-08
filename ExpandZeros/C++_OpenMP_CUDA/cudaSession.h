/*
Given a matrix, for any element with value 0,
set all elements from the corresponding row and column on 0, as well.

Implementations using OpenMP and CUDA for NVIDIA GPUs.

@2017 Florin Tulba (florintulba@yahoo.com)
*/

#ifndef H_CUDA_SESSION
#define H_CUDA_SESSION

#include <vector>
#include <functional>
#include <sstream>
#include <cassert>

#include <cuda_runtime.h>

/// @return x, so that reference * k == x and k = ceiling(val/reference)
template<size_t reference>
__host__ __device__ size_t nextMultipleOf(size_t val) {
	const size_t rest = val % reference;

	return (rest > 0ULL) ? (val + reference - rest) : val;
}

// Macro for checking the execution of a CUDA instruction
// When not successful, it throws runtime_error providing an explanation
#define CHECK_CUDA_OP(op) \
	{ \
		const cudaError_t errCode = op; \
		if(errCode != cudaSuccess) { \
			std::ostringstream oss; \
			oss<<"Operation " #op " failed because: "<<cudaGetErrorString(errCode); \
			throw runtime_error(oss.str()); \
		} \
	}

// Macro for checking if the launch of a CUDA kernel succeeded.
// When not successful, it throws runtime_error providing an explanation
#define CHECK_CUDA_KERNEL_LAUNCH(kernelName) \
	{ \
		const cudaError_t errCode = cudaGetLastError(); \
		if(errCode != cudaSuccess) { \
			std::ostringstream oss; \
			oss<<"The launch of kernel " #kernelName " failed because: " \
				<<cudaGetErrorString(errCode); \
			throw runtime_error(oss.str()); \
		} \
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

	CudaRAII(const CudaRAII&) = delete;
	CudaRAII(CudaRAII&&) = delete;
	void operator=(const CudaRAII&) = delete;
	void operator=(CudaRAII&&) = delete;

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

/// Sets up a CUDA session and preallocates device memory and streams
class CudaSession {
protected:
	void* reservedDevMem = nullptr;	///< memory area preallocated on the device
	std::vector<cudaStream_t> reservedStreams; ///< user preallocated streams

public:
	CudaSession(); ///< sets the current device

	/// Reserves sz B in device memory if reservedDevMem == nullptr. Otherwise throws logic_error
	void* reserveDevMem(size_t sz);

	/// Creates a new stream with given flags and priority
	cudaStream_t createStream(unsigned flags = cudaStreamDefault, int priority = 0);

	/// Destroys all streams
	void destroyStreams();

	/// Releases the reserved device memory
	void releaseDevMem();

	/// @return the available streams
	const std::vector<cudaStream_t>& getReservedStreams() const;

	/// @return the reserved memory, if any, reinterpreting it as char*
	char* getReservedMem() const;

	/**
	Releases any device memory and user streams and calls cudaDeviceReset().

	cudaDeviceReset() must be called before exiting in order for profiling and
	tracing tools such as Nsight and Visual Profiler to show complete traces.
	*/
	~CudaSession();
};

#endif // H_CUDA_SESSION
