#! python3

'''
Counting all possible triangles and (convex) quadrilaterals from geometric figures traversed by a number of lines.
See the tested figures in 'TestFigures/'.

Required modules:
 - <bitarray> - https://pypi.python.org/pypi/bitarray

@2017 Florin Tulba (florintulba@yahoo.com)
'''

import sys

# 'util.py' from '../common' contains the necessary function 'nextRelevantLine'
sys.path.append('../common')
from util import nextRelevantLine

from bitarray import bitarray

class ShapeCounter:
	''' Counts triangles and convex quadrilaterals from a figure '''

	bit1 = bitarray('1')	# static value used below

	def __init__(self, lines):
		'''
		Configures the ShapeCounter based on the sequences of named points from the lines from the figure.
		Prepares the entire infrastructure needed while counting the shapes.
		'''
		self.N = 0 						# Points Count
		self.L = len(lines)				# Lines Count
		self.triangles = 0 				# Count of triangles
		self.convexQuadrilaterals = 0 	# Count of convex quadrilaterals
		self.pointNames = []			# the names of the points
		self.lineMembers = []			# the indices of the members of each line (list of bitarray-s)
		self.connections = []			# points connectivity matrix (list of bitarray-s)
		self.membership = []			# membership of points to the lines (list of bitarray-s)
		self.membershipAsRanks = []		# for each point a map between lineIdx and rankWithinLine (list of dictionaries with entries like <lineIdx> : <pointRankWithinLine>)

		pointsIndices = {}
		for lineIdx in range(self.L):
			line = lines[lineIdx]
			pointsOnLine = len(line)
			memberIndices = []
			memberIndicesBitset = bitarray(self.N * '0')
			for pointRank in range(pointsOnLine):
				pointName = line[pointRank]
				if not pointName in pointsIndices:
					self.pointNames.append(pointName)
					pointIdx = self.N
					pointsIndices[pointName] = pointIdx
					self.N += 1
					for prevMemIdx in range(len(self.lineMembers)):
						self.lineMembers[prevMemIdx].append(False)
					for prevConnIdx in range(len(self.connections)):
						self.connections[prevConnIdx].append(False)
					self.connections.append(bitarray(self.N * '0'))
					memberIndicesBitset.append(True)
					self.membership.append(bitarray(self.L * '0'))
					self.membershipAsRanks.append({})
				else:
					pointIdx = pointsIndices[pointName]
					memberIndicesBitset[pointIdx] = True
				lastPointConns = self.connections[pointIdx]
				for prevMemIdx in range(len(memberIndices)):
					lastPointConns[memberIndices[prevMemIdx]] = True
					self.connections[memberIndices[prevMemIdx]][pointIdx] = True
				memberIndices.append(pointIdx)
				self.membership[pointIdx][lineIdx] = True
				self.membershipAsRanks[pointIdx][lineIdx] = pointRank
			self.lineMembers.append(memberIndicesBitset)

		'''
		# Uncomment if interested in inspecting the correctness of the configuration based on the current <lines> parameter
		for i in range(self.N):
			print('{}: connections {} ; member of lines {} ; pos in lines {}'.
				format(self.pointNames[i], self.connections[i], self.membership[i],
					[str(self.membershipAsRanks[i][lineIdx])+'(l'+str(lineIdx)+')'
						for lineIdx in self.membershipAsRanks[i].keys()]))
		for i in range(self.L):
			print('L{}: members {}'.format(i, self.lineMembers[i]))
		'''

	def process(self):
		''' Performs the actual shape counting '''
		# One step for ensuring the uniqueness of the solutions:
		# a mask to prevent the shapes found later from using points before P1.
		maskP1 = bitarray(self.N * '1')
		for p1 in range(self.N - 2):
			mem1 = self.membership[p1]
			maskP1[p1] = False # Ignore connections before and including P1
			connOfP1Bitset = self.connections[p1] & maskP1
			connOfP1 = connOfP1Bitset.search(ShapeCounter.bit1)
			countConnOfP1 = len(connOfP1)
			if countConnOfP1 < 2:
				continue # Triangles require 2 connected points to P1. If they are not available, check next available P1

			for idxP2, p2 in enumerate(connOfP1[ : countConnOfP1-1]):
				mem2 = self.membership[p2]
				mem1and2 = mem1 & mem2
				for idxLastP in range(idxP2 + 1, countConnOfP1):
					lastP = connOfP1[idxLastP]
					memLast = self.membership[lastP]
					if (mem1and2 & memLast).any(): # coll(p1, p2, lastP)
						continue	# Ignore collinear points

					if self.connections[p2][lastP]:
						self.triangles += 1
						print('<{}{}{}> '.format(self.pointNames[p1], self.pointNames[p2], self.pointNames[lastP]), end='')

					connOfP2_LastP_Bitset = self.connections[p2] & self.connections[lastP] & maskP1
					mem1and2or2andLast = mem1and2 | (mem2 & memLast)
					for _, p3 in enumerate(connOfP2_LastP_Bitset.search(ShapeCounter.bit1)):
						mem3 = self.membership[p3]
						if (mem1and2or2andLast & mem3).any(): # coll(p1, p2, p3) || coll(lastP, p2, p3)
							continue	# Ignore collinear points

						if self.convex(p1, mem1, p2, mem2, p3, mem3, lastP, memLast):
							self.convexQuadrilaterals += 1
							print('[{}{}{}{}] '.format(self.pointNames[p1], self.pointNames[p2], self.pointNames[p3], self.pointNames[lastP]), end='')
		print()

	def convex(self, p1, mem1, p2, mem2, p3, mem3, p4, mem4):
		''' Checks convexity of p1-p4 quadrilateral, based on the membership of each point to the available lines '''
		# Extended p1-p2 and p3-p4 shouldn't touch
		if not self.allowedIntersection([p1, p2], [p3, p4], (mem1 & mem2), mem3, mem4):
		   return False

		# Extended p2-p3 and p4-p1 shouldn't touch
		if not self.allowedIntersection([p2, p3], [p4, p1], (mem2 & mem3), mem4, mem1):
			return False

		return True

	def allowedIntersection(self, l1, l2, memL1, memL2_1, memL2_2):
		'''
		Returns true only if the extended lines l1 and l2 don't intersect, or intersect strictly outside the shape described by the 4 points from l1 and l2.
		Parameters:
			l1 		- one line from a potential quadrilateral
			l2 		- the line across from l1 in the potential quadrilateral
			memL1 	- 'and'-ed memberships (which lines include each point) of the 2 points from l1
			memL2_1 - membership (which lines include the point) of one point from l2
			memL2_1 - membership (which lines include the point) of the other point from l2
		'''
		lineIdxPair1 = memL1.search(ShapeCounter.bit1)[0]
		if memL2_1[lineIdxPair1] or memL2_2[lineIdxPair1]:
			return False # one of the provided points from L2 are inside L1

		lineIdxPair2 = (memL2_1 & memL2_2).search(ShapeCounter.bit1)[0]
		intersectionPoint = (self.lineMembers[lineIdxPair1] & self.lineMembers[lineIdxPair2]).search(ShapeCounter.bit1)
		if intersectionPoint:
			intersectionPoint = intersectionPoint[0]
			# The found intersection point should fall outside the segment l1
			# The check relies on the fact that lines specify the contained points in order
			rank1 = self.membershipAsRanks[l1[0]][lineIdxPair1]
			rank2 = self.membershipAsRanks[l1[1]][lineIdxPair1]
			if rank1 > rank2:
				rank1, rank2 = rank2, rank1
			intersectionPointMembership = self.membershipAsRanks[intersectionPoint]
			rank = intersectionPointMembership[lineIdxPair1]
			if rank1 <= rank <= rank2:
				return False

			# The found intersection point should fall outside the segment l2
			rank1 = self.membershipAsRanks[l2[0]][lineIdxPair2]
			rank2 = self.membershipAsRanks[l2[1]][lineIdxPair2]
			if rank1 > rank2:
				rank1, rank2 = rank2, rank1
			rank = intersectionPointMembership[lineIdxPair2]
			if rank1 <= rank <= rank2:
				return False

		return True # no intersection or the intersection is 'external'


if __name__ == "__main__":
	# Select one of the following scenario files (uncommenting it and commenting the others)
	# or create a new scenario and open your own file:


	# Scenario from figure 'count6Shapes.png'
	f = open('TestFigures/TextVersions/count6shapes.txt', 'r') # Manually generated (same labels as in the corresponding figure)
	#f = open('TestFigures/TextVersions/count6shapes.png.txt', 'r') # Generated by the figure interpreter written in Matlab/Octave

	# Scenario from figure 'count9Shapes.png'
	#f = open('TestFigures/TextVersions/count9shapes.txt', 'r') # Manually generated (same labels as in the corresponding figure)
	#f = open('TestFigures/TextVersions/count9shapes.png.txt', 'r') # Generated by the figure interpreter written in Matlab/Octave

	# Scenario from figure 'count100Shapes.png'
	#f = open('TestFigures/TextVersions/count100shapes.txt', 'r') # Manually generated (same labels as in the corresponding figure)
	#f = open('TestFigures/TextVersions/count100shapes.png.txt', 'r') # Generated by the figure interpreter written in Matlab/Octave

	# Scenario from figure 'count673Shapes.png'
	#f = open('TestFigures/TextVersions/count673shapes.txt', 'r') # Manually generated (same labels as in the corresponding figure)
	#f = open('TestFigures/TextVersions/count673shapes.png.txt', 'r') # Generated by the figure interpreter written in Matlab/Octave

	
	lines = []
	line = nextRelevantLine(f)
	while line != None:
		tokens = line.split()
		lines.append(tokens)
		line = nextRelevantLine(f)
	f.close()

	sc = ShapeCounter(lines)
	sc.process()
	totalShapes = sc.triangles + sc.convexQuadrilaterals
	print("There are", sc.triangles, "triangles and", sc.convexQuadrilaterals, "convex quadrilaterals, which means", totalShapes, "convex shapes in total.")
