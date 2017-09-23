/*
Finding the most distant pair of sorted elements in an array.

Compiled with g++ 5.4.0:
	g++ -std=c++14 -Ofast -Wall "../common/util.cpp" furthestSortedPair.cpp -o furthestSortedPair

@2017 Florin Tulba (florintulba@yahoo.com)
*/

#include "../common/util.h"

#include <vector>
#include <numeric>
#include <random>
#include <algorithm>
#include <iterator>
#include <iostream>
#include <cassert>

using namespace std;

/**
Slow, but safe method with O(N^2) for finding the furthest pair of sorted values
within vals.

It uses a main loop to consider all possible left members of the pair.
There is also an inner loop that checks for corresponding right members of the
pair starting from the right end of the array and stopping when the pair spacing
is better / inferior than the previous best.
*/
template<class Cont>
size_t referenceResult(const Cont &vals) {
	typedef typename Cont::value_type Type;
	const size_t valsCount = vals.size();
	if(0ULL == valsCount)
		return 0ULL;

	size_t maxDist = 0ULL;
	auto itLeft = begin(vals);
	for(size_t left = 0ULL; left + maxDist + 1ULL < valsCount; ++left, ++itLeft) {
		const Type &leftVal = *itLeft;
		auto itRight = rbegin(vals);
		for(size_t right = valsCount - 1ULL; right > left + maxDist; --right, ++itRight) {
			if(leftVal < *itRight) {
				maxDist = right - left;
				break;
			}
		}
	}
	return maxDist;
}

/// Details about key elements observed while traversing the array from the right
template<typename Type>
struct Info {
	const Type &val;	///< a newly detected maximum value
	const size_t right;	///< the index from the right where the value was found
	
	/// how many relevant items towards left rely on this value
	size_t coverage;

	Info(const Type &val_, size_t right_, size_t coverage_ = 1ULL) :
		val(val_), right(right_), coverage(coverage_) {}

	/// Eases referring to val even by upper_bound
	operator const Type& () const { return val; }
};

/**
The key elements observed while traversing the array from the right, which are
the only options for right pair members.
*/
template<typename Type>
class RightOptions {
	vector<Info<Type>> data;	///< the key elements with local maximum values
	typename vector<Info<Type>>::iterator backIt;	///< replacement of back() to avoid pop_back()

	bool addAllowed = true;	///< prevent adding after calling doneAdding()

	void validate() const { if(empty()) throw range_error("No options!"); }

public:
	/// Reserving data for worst case scenario
	RightOptions(size_t capacity) { data.reserve(capacity); backIt = end(data); }

	bool empty() const { return backIt == end(data); }

	size_t size() const {
		if(empty())
			return 0ULL;
		return 1ULL + (size_t)distance(cbegin(data),
			typename vector<Info<Type>>::const_iterator(backIt));
	}

	/// Registers a newly found maximum value
	void addNew(const Type &val, size_t right, size_t coverage = 1ULL) {
		if(!addAllowed) throw logic_error("Cannot add after calling doneAdding()!");
		data.emplace_back(val, right, coverage);
		backIt = --end(data);
	}

	/// Sets addAllowed on false
	void doneAdding() { addAllowed = false; }

	/// Last introduced option
	Info<Type>& lastKnown() const { validate(); return *backIt; }

	/// Returns the first option larger than val. val must be less than the largest option in data
	const Info<Type>& optionFor(const Type &val) const {
		validate();
		assert(val < *backIt);
		return *upper_bound(cbegin(data),
			next(typename vector<Info<Type>>::const_iterator(backIt)), val);
	}

	/// Withdraw coverage from latest options until reaching count or until data gets empty
	void reduce(size_t count = 1ULL) {
		while(count > 0ULL && !data.empty()) {
			const size_t coverageOfLast = backIt->coverage,
					removableCoverage = min(count, coverageOfLast);
			if(removableCoverage == coverageOfLast) {
				if(backIt != begin(data))
					--backIt;
				else
					backIt = end(data);
			} else
				backIt->coverage -= removableCoverage;
			count -= removableCoverage;
		}
	}
};

/**
This approach of finding the furthest pair of sorted values within vals works
in O(N*log(N)).

@param vals the array of elements
@param comparesCount overestimated value of the number of required compare operations

@return the distance between the furthest pair of sorted values of the array

It also uses a main loop to consider all possible left members of the pair.
However it considers the following facts:

- larger left members of the pair than previously considered cannot deliver
	better result. This means skipping left pair members larger than the
	minimum left pair member previously analyzed

- similarly, for a given left pair member, the corresponding right pair member
	might be strictly one of the (updated) maximum values found while traversing
	the array from right towards left

- the analysis of the right pair member for the first possible left pair member
	(the very first loop) allows improving the next passes. So, the index of each
	newly encountered maximum value (right to left traversal) can be recorded.

- the search for the first larger value than left pair member within the
	previously mentioned array of stored maximum values (recorded in ascending
	order) can be performed using a binary search, obtaining a log(N) for the
	inner loop
*/
template<class Cont>
size_t improvedMethod(const Cont &vals, size_t &comparesCount) {
	const size_t valuesCount = vals.size();
	comparesCount = 0ULL;

	if(valuesCount < 2ULL)
		return 0ULL; // There's no pair of elements yet

	const size_t lastIdx = valuesCount - 1ULL;
	auto itLeft = begin(vals);
	auto itRight = rbegin(vals);
	++comparesCount;
	if(*itLeft < *itRight)
		return lastIdx;

	size_t maxDist = 0ULL;

	typedef typename Cont::value_type Type;

	// First inspection of the array considering first element as the left pair member
	size_t left/* = 0ULL*/;
	RightOptions<Type> rightOptions(lastIdx);
	rightOptions.addNew(*itRight, lastIdx, 0ULL); // let coverage field be set to 1 within the loop
	for(size_t right = lastIdx; right > /*left + */maxDist; --right, ++itRight) {
		++comparesCount;
		if(*itRight > rightOptions.lastKnown()) {
			// new maximum, which can be then compared against *itLeft
			++comparesCount;
			if(*itLeft < *itRight) {
				maxDist = right/* - left*/;
				break;
			}

			// add the new max only if this matters when left=1
			rightOptions.addNew(*itRight, right);

		} else
			++rightOptions.lastKnown().coverage;
	}
	rightOptions.doneAdding();

	// Checking all remaining potential left pair members
	auto itMinVal = itLeft; // init min
	for(left = 1ULL, ++itLeft; left + maxDist < lastIdx; ++left, ++itLeft) {
		rightOptions.reduce(); // a value should be popped out for each iteration

		// assessing only local minimum left pair members
		++comparesCount;
		if(*itLeft >= *itMinVal)
			continue;

		itMinVal = itLeft; // renew min

		// Check if there is no right pair member
		assert(!rightOptions.empty());
		++comparesCount;
		if(*itLeft >= rightOptions.lastKnown())
			continue;

		// The appropriate right pair member can be found using binary search
		comparesCount += (size_t)ceil(log2(rightOptions.size())); // overestimate of the compare ops performed
		const Info<Type> &rightInfo = rightOptions.optionFor(*itLeft);

		// Discarding (rightInfo.right - left) - maxDist elements from rightOptions (if there are so many)
		// and assigning rightInfo.right - left to maxDist
		const size_t newMaxDist = rightInfo.right - left;
		rightOptions.reduce(newMaxDist - maxDist);
		maxDist = newMaxDist;
	}

	// At this point, rightOptions should be empty, or contain one option with coverage 1
	assert(rightOptions.empty() ||
		(rightOptions.size() == 1ULL && rightOptions.lastKnown().coverage == 1ULL));

	return maxDist;
}

/// Compares the results of the 2 approaches and reports errors and compare operations count
template<class Cont>
size_t checkUseCase(const Cont &vals, size_t &errorsCount, bool verbose = false) {
	typedef typename Cont::value_type Type;
	size_t comparesCount = 0ULL;
	const size_t refRes = referenceResult(vals), res = improvedMethod(vals, comparesCount);
	if(refRes != res) {
		++errorsCount;
		cerr<<"For the array below, the expected result was "<<refRes
			<<", but obtained "<<res<<" instead."<<endl;
		copy(CBOUNDS(vals), ostream_iterator<const Type&>(cerr, ", "));
		cerr<<"\b\b "<<endl<<endl;
	} else if(verbose) {
		cout<<"Furthest sorted pair of elements is at a distance of "<<res
			<<" in the array from below. It needed "<<comparesCount<<" compare ops."<<endl;
		copy(CBOUNDS(vals), ostream_iterator<const Type&>(cout, ", "));
		cout<<"\b\b "<<endl<<endl;
	}

	return comparesCount;
}

int main() {
	enum {TIMES = 1000,			// count of random arrays to be checked
		VALUES_COUNT = 1000};	// size of each random array

	size_t errorsCount = 0ULL;

	// The container should accept bidirectional iterators and
	// may contain any type that is comparable and can be displayed with <<
	typedef int Type;
	vector<Type> vals;

	// Empty array case
	checkUseCase(vals, errorsCount);

	// Single element array case
	vals.push_back(100);
	checkUseCase(vals, errorsCount);

	vals.resize(VALUES_COUNT);

	// Sorted array case
	iota(BOUNDS(vals), 0);
	checkUseCase(vals, errorsCount, true);

	// Descending sorted array case
	iota(rbegin(vals), rend(vals), 0);
	checkUseCase(vals, errorsCount, true);

	// Random arrays cases
	cout<<"Checking random arrays ..."<<endl;
    mt19937 g;
    shuffle(BOUNDS(vals), g);
	checkUseCase(vals, errorsCount, true);

	for(int t = 0; t < TIMES; ++t) {
		shuffle(BOUNDS(vals), g);
		checkUseCase(vals, errorsCount);
	}

/**/
	// Searching for the worst case scenario within the 3628800 permutations of an array of 10 elements
	cout<<"Looking for the worst case scenario ..."<<endl;
	size_t maxComparesCount = 0ULL;
	vector<Type> worstConfig(10);
	vals.resize(10);
	iota(BOUNDS(vals), 0);
	do {
		const size_t comparesCount = checkUseCase(vals, errorsCount);
		if(comparesCount > maxComparesCount) {
			maxComparesCount = comparesCount;
			worstConfig = vals;
		}
	} while(next_permutation(BOUNDS(vals)));
	cout<<"The worst configuration from below produced "<<maxComparesCount<<" compare ops."<<endl;
	copy(CBOUNDS(worstConfig), ostream_iterator<const Type&>(cout, ", "));
	cout<<"\b\b "<<endl<<endl;
/**/

	// Inspecting the worst case scenario found above:
	vals = { 9, 8, 7, 6, 5, 4, 3, 2, 1, 0 };
	size_t comparesCount = 0ULL;
	improvedMethod(vals, comparesCount);
	cout<<"The worst case scenario required "<<comparesCount<<" compare ops."<<endl;

	if(errorsCount > 0ULL)
		cerr<<"There were "<<errorsCount<<" errors!"<<endl;
	else
		cout<<"There were no errors"<<endl;

	return 0;
}