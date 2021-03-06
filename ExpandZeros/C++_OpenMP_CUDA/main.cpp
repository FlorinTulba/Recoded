/*
Given a matrix, for any element with value 0,
set all elements from the corresponding row and column on 0, as well.

Usually the results appear as sparse matrices.
It would make sense:

- (A) either to return just the sequence of the remaining non-zero elements (sparse-matrix format)
- (B) or to copy those non-zero remaining elements into a new matrix (dense-matrix format)

instead of (C) setting on zero probably way more elements from the affected rows / columns.

However, only solution (C) is in-place, which is the reason why it was chosen.

Implementations using OpenMP and CUDA for NVIDIA GPUs.

@2017 Florin Tulba (florintulba@yahoo.com)
*/

#include "expandZerosCUDA.h"
#include "expandZerosOpenMP.h"
#include "align.h"
#include "../../common/util.h"
#include "../../common/config.h"

#include <vector>
#include <iterator>
#include <fstream>
#include <random>

using namespace std;

const string& configFile() {
	static const string result("config.txt");
	return result;
}

const map<string, const unique_ptr<IConfigItem>>& Config::props() {
	static map<string, const unique_ptr<IConfigItem>> pairs;

#define addProp(propName, propType, propDefVal, validator) \
	pairs.emplace(#propName, make_unique<ConfigItem<propType>>(#propName, propDefVal, validator))

	// Add here all configuration items
	addProp(TIMES, size_t, 1000ULL, std::make_unique<InRange<size_t>>(1ULL, 10000000ULL));
	addProp(mMax, size_t, 500ULL, std::make_unique<InRange<size_t>>(16ULL, 10000ULL));
	addProp(nMax, size_t, 500ULL, std::make_unique<InRange<size_t>>(16ULL, 10000ULL));
	addProp(ZerosPercentage, double, .04, std::make_unique<InRange<double>>(1e-4, .9999));
	addProp(MinElemsPerOpenMPThread, size_t, 0ULL, nullptr);
	addProp(MinElemsForGPU, size_t, 0ULL, nullptr);
	addProp(WorkQuotaGPU, double, 1., std::make_unique<InRange<double>>(0., 1.));

#undef addProp

	return pairs;
}

namespace {
	random_device rd;
	mt19937 randGen(rd());

	/**
	Randomly initializes a matrix with elements in 0..1000 range.

	@param a the matrix to initialize
	@param m the number of rows of the matrix
	@param n the number of columns of the matrix
	@param dimA the number of elements in `a` [`checkRows`.size() * `checkCols`.size()]
	@param checkRows reports which rows contain elements equal to 0
	@param checkCols reports which columns contain elements equal to 0
	*/
	void initMat(int *a, size_t m, size_t n, size_t dimA,
				 vector<bool> &checkRows, vector<bool> &checkCols) {
		assert(nullptr != a && dimA > 0ULL && dimA == m * n);

		static const double origZerosPercentage =
			Config::get().valueOf(ConfigItem<double>("ZerosPercentage"), .04);
		static uniform_int_distribution<> uidVal(1, 1000);

		checkRows.clear(); checkRows.resize(m, false);
		checkCols.clear(); checkCols.resize(n, false);

		// Set all elements first on non-zero values
		for(size_t i = 0ULL; i<dimA; ++i)
			a[i] = uidVal(randGen);

		// Choose some coordinates to be set to 0
		uniform_int_distribution<size_t> uidPos(0ULL, dimA-1ULL);
		const size_t zerosCount = size_t(dimA * origZerosPercentage);
		for(size_t i = 0ULL; i<zerosCount; ++i) {
			const size_t pos = uidPos(randGen),
				row = pos / n, col = pos % n;
			a[pos] = 0;
			checkRows[row] = checkCols[col] = true;
		}
	}

	/**
	Reads a matrix from a stream.
	First line needs to contain `m` followed by `n`.
	Every following input line must contain the `n` elements from one row of the matrix.

	@param pa pointer towards the region of pinned memory allocated for the read matrix
	@param m the number of rows of the matrix
	@param n the number of columns of the matrix
	@param is the source stream of the matrix
	*/
	void inputMat(int **pa, size_t &m, size_t &n, istream &is = cin) {
		assert(nullptr != pa);
		string line;
		{ // Reading matrix size from the first line of the file: m followed by n
			if(!getline(is, line))
				throw runtime_error("Couldn't read the first line!");
			istringstream iss(line);
			if(!(iss>>m>>n))
				throw runtime_error("Couldn't read the size of the matrix!");
			CHECK_CUDA_OP(cudaMallocHost((void**)pa, m * n * sizeof(int))); // pinned memory
		}

		// Reading matrix data, row by row
		for(size_t r = 0ULL, idx = 0ULL; r < m; ++r) {
			if(!getline(is, line))
				throw runtime_error("Couldn't read a new line!");
			istringstream iss(line);
			for(size_t c = 0ULL; c < n; ++c, ++idx) {
				if(!(iss>>(*pa)[idx])) // Reading next element from the same row
					throw runtime_error("Couldn't read a matrix element!");
			}
		}
	}

	/**
	Outputs a given matrix to a stream.
	First line reports the matrix size: `m` followed by `n`.
	Every following input line provides the `n` elements from the corresponding row of the matrix.

	@param a the matrix to be presented
	@param m the number of rows of the matrix
	@param n the number of columns of the matrix
	@param os the destination stream for the matrix
	*/
	void outputMat(const int * const a, size_t m, size_t n, ostream &os = cout) {
		assert(nullptr != a);
		os<<m<<'\t'<<n<<endl;
		size_t idx = 0ULL;
		for(size_t r = 0ULL; r < m; ++r) {
			for(size_t c = 0ULL; c < n; ++c, ++idx)
				os<<setw(4)<<a[idx]<<" ";
			os<<endl;
		}
		os<<endl;
	}

	/// @return true if the GPU returns the expected indices for the rows / columns to reset
	bool same(const bool * const found, const vector<bool> &expected) {
		assert(nullptr != found && !expected.empty());

		if(equal(CBOUNDS(expected),
			stdext::make_checked_array_iterator(found, distance(CBOUNDS(expected)))))
			return true;

		cerr<<"Mismatch: "<<endl<<"\tExpected: ";
		copy(CBOUNDS(expected), ostream_iterator<bool>(cerr));

		cerr<<endl<<"\tReceived: ";
		for(size_t j = 0, len = expected.size(); j < len; ++j)
			cerr<<found[j];
		cerr<<endl<<endl;

		return false;
	}

	/// Resets the vector of found rows / columns for reprocessing by a different algorithm
	void clearFound(bool * const found, size_t count) {
		for(size_t i = 0ULL; i < count; ++i)
			found[i] = false;
	}
} // anonymous namespace

/// Compares the performances of the OpenMP vs CUDA implementations
void main() try {
	const Config &cfg = Config::get();		
	const size_t TIMES = cfg.valueOf(ConfigItem<size_t>("TIMES"), 1000ULL), // iterations count
		mMax = cfg.valueOf(ConfigItem<size_t>("mMax"), 500ULL), // max matrix height
		nMax = cfg.valueOf(ConfigItem<size_t>("nMax"), 500ULL), // max matrix width
		dimResults = (mMax + nMax) * sizeof(bool), // sum of the sizes of the arrays for found zeros on rows / columns
		maxElemsA = mMax*nMax, // max matrix elements
		dimAMax = maxElemsA * sizeof(int), // max size of the matrix
		dimAMaxPadded = nextMultipleOf<256ULL>(dimAMax); // padded size of the largest matrix

	uniform_int_distribution<size_t>
		mRand(15, mMax-1), nRand(15, nMax-1);

	// Aligned allocation helps preventing false sharing in the OpenMP algorithm version
	AlignedMemRAII<int> origMat(ArrayRequest(maxElemsA), l1CacheLineSz());
		
	char *hostMem = nullptr;
	CudaRAII<void*> pinnedHostMem(function<cudaError_t(void**, size_t)>(::cudaMallocHost),
									::cudaFreeHost,
									(void*&)hostMem,
									dimAMaxPadded + dimResults);

	int *a = (int*)hostMem;
	bool *results = (bool*)&hostMem[dimAMaxPadded],
		*foundCols = results; // foundCols is always at the beginning of the results
	// foundRows will be right after foundCols, which depends on the width of the matrix from each test

	// Testing TIMES matrices of random sizes and with random elements
	// The timers must ignore TIMES, as the size of the matrix is different for each iteration
	// and only the average required time per element does makes sense in this case
	Timer timerCUDA("timerCUDA", 1ULL, false), timerOpenMP("timerOpenMP", 1ULL, false);
	size_t totalElems = 0ULL; // Count of the analyzed elements of all matrices from all iterations
		
	vector<bool> checkRows, checkCols; // Correct rows / columns containing values of 0
	for(int i = 0; i < TIMES; ++i) {
		// Init random dimensions and matrix
		const size_t m = mRand(randGen),
			n = nRand(randGen),
			dimMat = m * n;
		totalElems += dimMat;
		initMat(origMat.get(), m, n, dimMat, checkRows, checkCols);

		// copy the original to be modified by the CUDA algorithm version
		memcpy_s((void*)a, dimAMax,
					(const void*)origMat.get(), dimMat * sizeof(int));

		// Expand the zeros from the matrix while timing the operation
		timerCUDA.resume();
		reportAndExpandZerosCUDA(a, (unsigned)m, (unsigned)n, results);
		timerCUDA.pause();

		bool *foundRows = &results[n];

		// Validate result
		assert(same(foundRows, checkRows));
		assert(same(foundCols, checkCols));

		// Test using OpenMP
		clearFound(results, m + n);

		// Expand the zeros directly from the original matrix while timing the operation
		timerOpenMP.resume();
		reportAndExpandZerosOpenMP(origMat.get(), (long)m, (long)n, foundRows, foundCols);
		timerOpenMP.pause();

		// Validate result
		assert(same(foundRows, checkRows));
		assert(same(foundCols, checkCols));
	}

	cout<<"Total time CUDA: "<<timerCUDA.elapsed()<<"s for "<<totalElems<<" elements in "
		<<TIMES<<" matrices, which means "<<timerCUDA.elapsed() * 1e9 / totalElems
		<<"ns/element"<<endl;
	cout<<"Total time OpenMP: "<<timerOpenMP.elapsed()<<"s for "<<totalElems<<" elements in "
		<<TIMES<<" matrices, which means "<<timerOpenMP.elapsed() * 1e9 / totalElems
		<<"ns/element"<<endl;

	timerCUDA.done();
	timerOpenMP.done();

} catch(exception &e) {
	/*
	e might be:
	- invalid_argument when instantiating Config
	- logic_error issued when violating preconditions or common sense facts
	- runtime_error thrown when a certain critical information cannot be obtained
		or when a critical operation cannot be performed
	*/
	cerr<<e.what()<<endl;
}
