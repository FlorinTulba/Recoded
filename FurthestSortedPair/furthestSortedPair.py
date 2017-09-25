#! python3

'''
Determine the most distant pair of sorted elements within a given array

@2017 Florin Tulba (florintulba@yahoo.com)

'''

import sys
import random
import math
from itertools import permutations
from bisect import bisect_right

def referenceResult(vals):
	'''
	Slow, but safe method with O(N^2) for finding the furthest pair of sorted values
	within vals.

	It uses a main loop to consider all possible left members of the pair.
	There is also an inner loop that checks for corresponding right members of the
	pair starting from the right end of the array and stopping when the pair spacing
	is better / inferior than the previous best.
	'''
	valsCount = len(vals)
	if valsCount == 0:
		return 0

	maxDist, left = 0, 0
	while left + maxDist + 1 < valsCount:
		leftVal = vals[left]
		right = valsCount - 1
		while right > left + maxDist:
			if leftVal < vals[right]:
				maxDist = right - left
				break
			right -= 1
		left += 1

	return maxDist

def improvedMethod(vals, comparesCount):
	'''
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
	'''
	valuesCount = len(vals)
	comparesCount[0] = 0
	if valuesCount < 2:
		return 0 # There's no pair of elements yet

	lastIdx = valuesCount - 1
	left, leftVal, right, rightVal = 0, vals[0], lastIdx, vals[-1]
	comparesCount[0] += 1
	if leftVal < rightVal:
		return lastIdx

	maxDist = 0

	# First inspection of the array considering first element as the left pair member
	maxValues, indices, coverages = [rightVal], [right], [0] # let coverage field be set to 1 within the loop
	ptrArrays = 0 # maintains the current position within previous lists
	while right > maxDist:
		comparesCount[0] += 1
		if rightVal > maxValues[-1]:
			# new maximum, which can be then compared against leftVal
			comparesCount[0] += 1
			if leftVal < rightVal:
				maxDist = right
				break

			# add the new max only if this matters when left=1
			maxValues.append(rightVal)
			indices.append(right)
			coverages.append(1)
			ptrArrays += 1

		else:
			coverages[-1] += 1

		right -= 1
		rightVal = vals[right]

	# Checking all remaining potential left pair members
	minVal = leftVal; # init min
	left = 1
	while left + maxDist < lastIdx:
		leftVal = vals[left]

		# a value should be popped out for each iteration
		if coverages[ptrArrays] > 1:
			coverages[ptrArrays] -= 1
		else:
			ptrArrays -= 1

		# assessing only local minimum left pair members
		comparesCount[0] += 1
		if leftVal >= minVal:
			left += 1
			continue

		minVal = leftVal # renew min

		# Check if there is no right pair member
		assert ptrArrays >= 0
		comparesCount[0] += 1
		if leftVal >= maxValues[ptrArrays]:
			left += 1
			continue

		# The appropriate right pair member can be found using binary search
		comparesCount[0] += math.ceil(math.log2(ptrArrays+1)) # overestimate of the compare ops performed
		idxOfLargerVal = bisect_right(maxValues[:ptrArrays+1], leftVal);

		# Discarding (indices[idxOfLargerVal] - left) - maxDist elements 
		# from rightOptions (if there are so many)
		# and assigning indices[idxOfLargerVal] - left to maxDist
		newMaxDist = indices[idxOfLargerVal] - left;

		toRemove = newMaxDist - maxDist
		while toRemove > 0 and ptrArrays >= 0:
			avail = coverages[ptrArrays]
			removing = min(avail, toRemove)
			if avail == removing:
				ptrArrays -= 1
			else:
				coverages[ptrArrays] -= removing
			toRemove -= removing

		maxDist = newMaxDist

		left += 1

	# At this point, rightOptions should be empty,
	# or contain one option with coverage 1
	assert ptrArrays < 0 or (ptrArrays == 0 and coverages[0] == 1)

	return maxDist

def checkUseCase(vals, errorsCount, verbose = False):
	'''
	Compares the results of the 2 approaches and reports errors
	and compare operations count
	'''
	comparesCount = [0]

	refRes = referenceResult(vals)
	res = improvedMethod(vals, comparesCount)
	if refRes != res:
		errorsCount[0] += 1
		print("For the array below, the expected result was", refRes,
			", but obtained", res, "instead.", file=sys.stderr)
		print(vals, file=sys.stderr)
		print()

	elif verbose:
		print("Furthest sorted pair of elements is at a distance of", res,
			"in the array from below. It needed", comparesCount[0],
			"compare ops.")
		print(vals)
		print()

	return comparesCount[0]

if __name__ == "__main__":
	TIMES = 1000			# count of random arrays to be checked
	VALUES_COUNT = 1000		# size of each random array

	errorsCount = [0]

	vals = []	# the array containing the values for the tests
	# Empty array case
	checkUseCase(vals, errorsCount)

	# Single element array case
	vals.append(100)
	checkUseCase(vals, errorsCount)

	# Sorted array case
	vals = [i for i in range(0, VALUES_COUNT)]
	checkUseCase(vals, errorsCount, True)

	# Descending sorted array case
	vals = [i for i in range(VALUES_COUNT-1, -1, -1)]
	checkUseCase(vals, errorsCount, True)

	# Random arrays cases
	print("Checking random arrays ...")
	random.shuffle(vals)
	checkUseCase(vals, errorsCount, True)

	for t in range(0, TIMES):
		random.shuffle(vals)
		checkUseCase(vals, errorsCount)

	# Searching for the worst case scenario within the 3628800 permutations
	# of an array of 10 elements
	print("Looking for the worst case scenario ...")
	maxComparesCount = 0
	worstConfig = []
	vals = [i for i in range(0, 10)]
	for perm in permutations(vals):
		comparesCount = checkUseCase(perm, errorsCount)
		if comparesCount > maxComparesCount:
			maxComparesCount = comparesCount
			worstConfig = perm
	print("The worst configuration from below produced", maxComparesCount, 
		"compare ops.")
	print(worstConfig)
	print()

	# Inspecting the worst case scenario found above:
	vals = [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
	comparesCount = [0]
	improvedMethod(vals, comparesCount)
	print("The worst case scenario required", comparesCount[0], "compare ops.")

	if errorsCount[0] > 0:
		print("There were", errorsCount[0], "errors!", file=sys.stderr)
	else:
		print("There were no errors")
