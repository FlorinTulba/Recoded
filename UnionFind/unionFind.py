#! python3

#
# Implementation of the UnionFind data structure described here:
# 		https://en.wikipedia.org/wiki/Disjoint-set_data_structure
#
# @2017 Florin Tulba (florintulba@yahoo.com)
#


import sys

# 'util.py' from '../common' contains the necessary function 'nextRelevantLine'
sys.path.append('../common')
from util import nextRelevantLine


class UfItem:
	''' Data specific for each item of the Union Find '''
	def __init__(self, id):
		self.ancestor = id 	# the most distant known ancestor
		self.rank = 0		# initial depth of a subtree rooted on this item (some children might migrate closer to the root of this item)

class UF:
	''' The UnionFind class '''
	def __init__(self, itemsCount):
		if itemsCount < 0:
			print('The items count must be >= 0!')

		self.items = [UfItem(i) for i in range(itemsCount)]	# the managed items
		self.groups = itemsCount							# current count of the groups formed from the items
		
		print(" Initially:{}".format(self))

		if itemsCount < 2:
			print("Note that this problem makes sense only for at least 2 elements!")

	def __repr__(self):
		mapping = {}
		for i in range(len(self.items)):
			parentId = self.parentOf(i)
			if parentId in mapping:
				mapping[parentId].append(i)
			else:
				mapping[parentId] = [i]

		result = '{:3d} groups: {}'.format(self.groups, mapping)
		return result

	def parentOf(self, id):
		''' Find parent operation '''
		if id < 0 or id >= len(self.items):
			return None

		parentId = self.items[id].ancestor
		while id != parentId:
			self.items[id].ancestor = self.items[parentId].ancestor
			id = self.items[id].ancestor
			parentId = self.items[id].ancestor
		return id

	def join(self, id1, id2):
		''' Connect id1 & id2 '''
		print("{:3d} - {:3d} :".format(id1, id2), end='')
		id1, id2 = self.parentOf(id1), self.parentOf(id2)
		if id1 == None or id2 == None:
			print('Invalid indices!')

		if id1 == id2:
			print(self)
			return

		rank1, rank2 = self.items[id1].rank, self.items[id2].rank
		if rank1 < rank2:
			self.items[id1].ancestor = id2
		else:
			self.items[id2].ancestor = id1

		if rank1 == rank2:
			self.items[id1].rank += 1

		self.groups -= 1

		print(self)

if __name__ == "__main__":
	f = open('testScenario.txt', 'r')
	line = nextRelevantLine(f)
	if line != None:
		n = int(line)
		uf = UF(n)
		line = nextRelevantLine(f)
		while line != None:
			tokens = line.split()
			if len(tokens) != 2:
				print("Line {} contains less/more than 2 items!".format(line))
				break
			idx1, idx2 = int(tokens[0]), int(tokens[1])
			uf.join(idx1, idx2)

			line = nextRelevantLine(f)

		if line == None:
			print("No other pairs to connect in this scenario!")			
	else:
		print("Couldn't read the items count!")

	f.close()
