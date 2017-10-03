#include "align.h"

// <new> is supposed to contain `hardware_destructive_interference_size` starting from C++17
#include <new>

// <Windows.h> is used when `hardware_destructive_interference_size` is not defined
#include <Windows.h>

#include <memory>
#include <sstream>

using namespace std;

#define CHECK_WIN_OP(op, expectedResult) \
	if((FALSE == op) && (expectedResult != GetLastError())) { \
		ostringstream oss; \
		oss<<#op " failed with error "<<GetLastError(); \
		throw runtime_error(oss.str()); \
	}

size_t l1CacheLineSz() {
	// `hardware_destructive_interference_size` (available from C++17) should provide the result
	__if_exists(::hardware_destructive_interference_size) {
		return ::hardware_destructive_interference_size;
	}
	__if_not_exists(::hardware_destructive_interference_size) {
		// Code based on https://github.com/NickStrupat/CacheLineSize
		size_t result = 0ULL;

		DWORD bytesInfo = 0UL;
		CHECK_WIN_OP(GetLogicalProcessorInformation(nullptr, &bytesInfo), ERROR_INSUFFICIENT_BUFFER);
		
		const size_t valuesCount = bytesInfo / sizeof(SYSTEM_LOGICAL_PROCESSOR_INFORMATION);
		unique_ptr<SYSTEM_LOGICAL_PROCESSOR_INFORMATION[]> info = 
			make_unique<SYSTEM_LOGICAL_PROCESSOR_INFORMATION[]>(valuesCount);
		CHECK_WIN_OP(GetLogicalProcessorInformation(info.get(), &bytesInfo), ERROR_SUCCESS);

		for(size_t i = 0; i < valuesCount; ++i) {
			const SYSTEM_LOGICAL_PROCESSOR_INFORMATION &val = info[i];
			if(val.Relationship == RelationCache && val.Cache.Level == 1) {
				result = val.Cache.LineSize;
				break;
			}
		}

		return result;
	}
}
