#ifndef H_ALIGN
#define H_ALIGN

#include <malloc.h>

/// Returns the size of the cache line from the L1 cache
size_t l1CacheLineSz();

/**
Helps AlignedMemRAII constructors to differentiate themselves.
When allocating an array, the first parameter must be an ArrayRequest
instead of another confusing size_t value (representing the length of the desired array)
*/
class ArrayRequest {
protected:
	size_t len; ///< the size of the requested array

public:
	ArrayRequest(size_t dim = 0ULL) : len(dim) {}

	size_t dim() const { return len; }
};

/// Allocates aligned memory and ensures it gets released. C++17 already accepts alignment for `new`
template<class T>
class AlignedMemRAII {
protected:
	T * const p = nullptr;		///< the pointer to either a T, or an array of T-s
	ArrayRequest checkArray;	///< non-default initialized only for actual arrays

public:
	///< Creates an actual T on heap at an aligned address with `alignment`
	template<class ... ArgTypes>
	AlignedMemRAII(size_t alignment, ArgTypes ... args) :
			p((T*const)_aligned_malloc(sizeof(T), alignment)) {
		if(nullptr == p)
			throw std::bad_alloc();

		new ((void*)p) T(std::forward<ArgTypes>(args)...);
	}

	///< Allocates an array of default initialized T-s at an aligned address with `alignment`
	AlignedMemRAII(const ArrayRequest &arraySpec, size_t alignment) :
			p((T*const)_aligned_malloc(sizeof(T) * arraySpec.dim(), alignment)),
			checkArray(arraySpec) {
		if(nullptr == p)
			throw std::bad_alloc();

		if(checkArray.dim() == 0ULL) {
			_aligned_free(p);
			throw std::logic_error("Provided ArrayRequest for 0-length array");
		}
		
		new ((void*)p) T[checkArray.dim()];
	}
	
	AlignedMemRAII(const AlignedMemRAII&) = delete;
	AlignedMemRAII(AlignedMemRAII&&) = delete;
	void operator=(const AlignedMemRAII&) = delete;
	void operator=(AlignedMemRAII&&) = delete;
	
	~AlignedMemRAII() {
		if(checkArray.dim() == 0ULL)
			p->~T();
		else {
			for(size_t i = 0ULL, len = checkArray.dim(); i <len; ++i)
				p[i].~T();
		}
		_aligned_free(p);
	}

	T* get() const { return p; }

	const T& operator[](size_t idx) const {
		if(checkArray.dim() == 0ULL)
			throw std::logic_error("Should not index a non-array!");
		if(idx >= checkArray.dim())
			throw std::range_error("Invalid index!");
		return p[idx];
	}

	T& operator[](size_t idx) {
		if(checkArray.dim() == 0ULL)
			throw std::logic_error("Should not index a non-array!");
		if(idx >= checkArray.dim())
			throw std::range_error("Invalid index!");
		return p[idx];
	}
};

#endif // H_ALIGN
