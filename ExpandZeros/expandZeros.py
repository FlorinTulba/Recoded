#! python3

'''
Given a matrix, for any element with value 0,
set all elements from the corresponding row and column on 0, as well.

Required modules:
 - <bitarray> - https://pypi.python.org/pypi/bitarray

@2017 Florin Tulba (florintulba@yahoo.com)
'''

import sys
import random
import time
import math
import multiprocessing
from itertools import repeat
from bitarray import bitarray
from threading import Thread

# Matrix settings
mMax = 500 # max number of rows
nMax = 500 # max number of columns
dimAMax = mMax * nMax # max number of elements
origZerosPercentage = 4 # Desired percentage of zeros within generated matrices

# Threshold below which a thread should not be created
MinElemsPerThread = 200000 # minimum number of matrix elements to be processed by a created thread
processorsCount = multiprocessing.cpu_count()

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

	def buildColIndices(self):
		'''
		extract the indices of the columns containing value 0
		'''
		colIndices = []
		idx = -1
		while True:
			try: # Searching for the 1st True after idx
				idx = self.foundCols.index(True, idx + 1)
			except ValueError:
				break
			colIndices.append(idx)
		return colIndices

	def findZerosInChunk(self, chunkRowsRange):
		'''
		Detects the rows and columns containing the value 0
		within the provided chunk of data
		'''
		# Avoids false sharing of foundCols through a local vector
		localFoundCols = bitarray(self.n * '0')

		for r in chunkRowsRange:
			rowContains0 = False
			idx = r * self.n
			for c in range(self.n):
				if self.a[idx] == 0:
					localFoundCols[c], rowContains0 = True, True
				idx += 1
			if rowContains0:
				self.foundRows[r] = True

		# Perform an union of all localFoundCols into foundCols
		for c in range(self.n):
			if localFoundCols[c] and not self.foundCols[c]:
				'''
				Minimizes false sharing of foundCols by reducing writing operations.
				Overwriting events actually don't change the data,
				but they invalidate the corresponding L1 cache line from other threads.
				This is however cheaper than a synchronization mechanism.
				'''
				self.foundCols[c] = True

	class ZerosDetector(Thread):
		'''
		Thread detecting the rows and columns containing the value 0
		within the provided chunk of data
		'''
		def __init__(self, outer, chunkRowsRange):
			Thread.__init__(self)
			self.outer = outer
			self.chunkRowsRange = chunkRowsRange # the chunk of data between 2 rows

		def run(self):
			# Override
			self.outer.findZerosInChunk(self.chunkRowsRange)

	def expandZerosInChunk(self, colIndices, chunkRowsRange):
		'''
		Expands the zeros from the rows and columns containing an original 0
		in the provided chunk of data
		'''
		for r in chunkRowsRange:
			rowStart = r * self.n
			if self.foundRows[r]:
				self.a[rowStart : rowStart + self.n] = repeat(0, self.n)
			else:
				for colIdx in colIndices:
					self.a[rowStart + colIdx] = 0

	class ZerosExpander(Thread):
		'''
		Thread expanding the zeros from the rows and columns
		containing an original 0 in the provided chunk of data
		'''
		def __init__(self, outer, colIndices, chunkRowsRange):
			Thread.__init__(self)
			self.outer = outer
			self.colIndices = colIndices # the columns where original zeros were found
			self.chunkRowsRange = chunkRowsRange # the chunk of data between 2 rows

		def run(self):
			# Override
			self.outer.expandZerosInChunk(self.colIndices, self.chunkRowsRange)

	def apply(self):
		'''
		Expands all the zeros from the matrix on the corresponding rows and columns
		'''
		assert [] != self.a and self.m > 0 and self.n > 0 \
			and self.dim == self.m * self.n \
			and self.checkRows != None and self.checkCols != None

		self.foundRows = bitarray(self.m * '0')
		self.foundCols = bitarray(self.n * '0')

		'''
		Distribute several consecutive rows to every spawn thread and
		keep the remaining ones to be processed in this main thread.

		There is no point spawning a thread unless it has a minimum
		number of elements to analyze.
		
		The main thread should process most of the time fewer elements,
		as it also needs to wait for the created threads.
		'''
		chunkSz = int(math.ceil( \
			max(self.m / processorsCount, \
				MinElemsPerThread / self.n)))
		chunkStartRow, chunkPastLastRow = 0, chunkSz
		additionalThreads = int(math.ceil(self.m / chunkSz)) - 1
		threads = []
		while additionalThreads > 0:
			additionalThreads -= 1
			threads.append( \
				ExpandZeros.ZerosDetector(self, \
					range(chunkStartRow, chunkPastLastRow)))
			chunkStartRow = chunkPastLastRow
			chunkPastLastRow += chunkSz
		for t in threads:
			t.start()

		self.findZerosInChunk(range(chunkStartRow, self.m))

		for t in threads:
			t.join()

		colIndices = self.buildColIndices()

		chunkStartRow, chunkPastLastRow = 0, chunkSz
		additionalThreads = len(threads)
	
		while additionalThreads > 0:
			additionalThreads -= 1
			threads[additionalThreads] = \
				ExpandZeros.ZerosExpander(self, \
					colIndices, \
					range(chunkStartRow, chunkPastLastRow))
			chunkStartRow = chunkPastLastRow
			chunkPastLastRow += chunkSz
		for t in threads:
			t.start()

		self.expandZerosInChunk(colIndices, range(chunkStartRow, self.m))

		for t in threads:
			t.join()

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
	print("Processed", totalElems, "elements in", TIMES, "matrices in %.3fs" % elapsed);
	print("That is %.3fns / element." % (elapsed * 1e9 / totalElems));
