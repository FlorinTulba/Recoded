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
class KernelLaunchConfig {
protected:
	unsigned _blocksCount;		///< the number of blocks to execute the kernel
	unsigned _threadsPerBlock;	///< how many threads does a block contain
	unsigned _shMemSz;			///< the size of the shared memory for a block
	cudaStream_t _stream;		///< the stream where to execute the kernel

public:
	KernelLaunchConfig(unsigned blocksCount_ = 1U,
					   unsigned threadsPerBlock_ = 32U,
					   unsigned shMemSz_ = 0U,
					   cudaStream_t stream_ = nullptr);

	void setBlocksCount(unsigned blocksCount_);
	inline unsigned blocksCount() const { return _blocksCount; }

	void setThreadsPerBlock(unsigned threadsPerBlock_);
	inline unsigned threadsPerBlock() const { return _threadsPerBlock; }

	void setSharedMemSize(unsigned shMemSz_);
	inline unsigned sharedMemSize() const { return _shMemSz; }

	void setStream(cudaStream_t stream_);
	inline cudaStream_t stream() const { return _stream; }
};

/// Sets up a CUDA session and preallocates device memory and streams
class CudaSession {
protected:
	void* reservedDevMem = nullptr;	///< memory area preallocated on the device
	std::vector<cudaStream_t> reservedStreams; ///< user preallocated streams
	std::vector<cudaEvent_t> reservedEvents; ///< user preallocated events

public:
	CudaSession(); ///< sets the current device

	/// Reserves sz B in device memory if reservedDevMem == nullptr. Otherwise throws logic_error
	void* reserveDevMem(size_t sz);

	/// Creates a new stream with given flags and priority
	cudaStream_t createStream(unsigned flags = cudaStreamDefault, int priority = 0);

	/// Creates a new event with given flags
	cudaEvent_t createEvent(unsigned flags = cudaEventDefault);

	/// Releases the reserved device memory
	void releaseDevMem();

	/// Destroys all streams
	void destroyStreams();

	/// Destroys all events
	void destroyEvents();

	/// @return the reserved memory, if any, reinterpreting it as char*
	char* getReservedMem() const;

	/// @return the available streams
	const std::vector<cudaStream_t>& getStreamsPool() const;

	/// @return the available events
	const std::vector<cudaEvent_t>& getEventsPool() const;

	/**
	Releases any device memory, user streams and user events and calls cudaDeviceReset().

	cudaDeviceReset() must be called before exiting in order for profiling and
	tracing tools such as Nsight and Visual Profiler to show complete traces.
	*/
	~CudaSession();
};

#endif // H_CUDA_SESSION
