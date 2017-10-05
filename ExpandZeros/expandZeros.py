#! python3

'''
Given a matrix, for any element with value 0,
set all elements from the corresponding row and column on 0, as well.

Required modules:
 - <bitarray> - https://pypi.python.org/pypi/bitarray

@2017 Florin Tulba (florintulba@yahoo.com)
'''

import math
import random
import time
from itertools import repeat
from bitarray import bitarray

# Matrix settings
mMax = 500 # max number of rows
nMax = 500 # max number of columns
dimAMax = mMax * nMax # max number of elements
origZerosPercentage = 4 # Desired percentage of zeros within generated matrices

class ExpandZeros:
	'''
	Implements the algorithm to expand the zeros and provides code
	for testing it.
	'''
	def __init__(self):
		'''
		It only preallocates the matrix for max allowed size
		'''
		self.a = dimAMax * [0] # the linearized matrix, preallocated for max allowed
		self.m = 0		# the actual number of rows
		self.n = 0		# the actual number of columns
		self.dim = 0	# actual number of matrix elements
		self.checkRows = None # the correct rows containing value 0
		self.checkCols = None	# the correct columns containing value 0
		self.foundRows = None # the found rows containing value 0
		self.foundCols = None # the found columns containing value 0

	def randInitMat(self):
		'''
		Chooses the dimensions and the elements of matrix a and provides
		checkRows and checkCols
		'''
		self.m = random.randrange(15, mMax) # random between [15, mMax)
		self.n = random.randrange(15, nMax) # random between [15, nMax)
		self.dim = self.m * self.n
		self.checkRows = bitarray(self.m * '0')
		self.checkCols = bitarray(self.n * '0')

		# Set all elements first on non-zero values
		for i in range(self.dim):
			self.a[i] = random.randrange(1, 1000) # random between [1, 1000)

		# Choose some coordinates to be set to 0
		zerosCount = self.dim * origZerosPercentage // 100
		for i in range(zerosCount):
			pos = random.randrange(self.dim)
			row, col = pos // self.n, pos % self.n
			self.a[pos] = 0
			self.checkRows[row], self.checkCols[col] = True, True

	def correct(self):
		'''
		Signals any mismatches between (checkCols - foundCols) and (checkRows - foundRows)
		Returns True if there are no mismatches
		'''
		if None == self.checkRows or None == self.checkCols or \
			None == self.foundRows or None == self.foundCols:
			return True

		if self.checkRows != self.foundRows:
			print("Rows Mismatches:", file=sys.stderr)
			print("\tExpected:", self.checkRows, file=sys.stderr)
			print("\tReceived:", self.foundRows, file=sys.stderr)
			return False

		if self.checkCols != self.foundCols:
			print("Columns Mismatches:", file=sys.stderr)
			print("\tExpected: ", self.checkCols, file=sys.stderr)
			print("\tReceived: ", self.foundCols, file=sys.stderr)
			return False

		return True

	def apply(self):
		'''
		Expands all the zeros from the matrix on the corresponding rows and columns
		'''
		assert [] != self.a and self.m > 0 and self.n > 0 \
			and self.dim == self.m * self.n \
			and self.checkRows != None and self.checkCols != None

		# Detects the rows and columns containing the value 0
		# within the provided chunk of data		
		self.foundRows = bitarray(self.m * '0')
		self.foundCols = bitarray(self.n * '0')
		for r in range(self.m):
			rowContains0 = False
			idx = r * self.n
			for c in range(self.n):
				if self.a[idx] == 0:
					self.foundCols[c], rowContains0 = True, True
				idx += 1
			if rowContains0:
				self.foundRows[r] = True

		# pairs like (startColIdx, lenRange) based on self.foundCols
		colRanges = []
		idx1, idx0 = 0, -1
		while True:
			try: # Searching for the 1st True after idx0
				idx1 = self.foundCols.index(True, idx0 + 1)
			except ValueError:
				break

			try: # Looking for the 1st 0 after idx1
				idx0 = self.foundCols.index(True, idx1 + 1)
			except ValueError:
				colRanges.append([idx1, self.n - idx1])
				break

			colRanges.append([idx1, idx0 - idx1])
	
		# Expands the zeros from the rows and columns containing an original 0
		# in the provided chunk of data
		for r in range(self.m):
			rowStart = r * self.n
			if self.foundRows[r]:
				self.a[rowStart : rowStart + self.n] = repeat(0, self.n)
			else:
				for startColIdx, lenRange in colRanges:
					idx = rowStart + startColIdx
					self.a[idx : idx + lenRange] = repeat(0, lenRange)


if __name__ == "__main__":
	random.seed()
	elapsed, lastStart, totalElems = 0, 0, 0
	proc = ExpandZeros()
	TIMES = 10 # iterations count
	for i in range(TIMES):
		proc.randInitMat()
		totalElems += proc.dim
		lastStart = time.perf_counter()
		proc.apply()
		elapsed += time.perf_counter() - lastStart
		assert proc.correct()
	print("Processed", totalElems, "elements in", TIMES, "matrices in", elapsed / 1e9,"s");
	print("That is", elapsed / totalElems, "ns / element.");
