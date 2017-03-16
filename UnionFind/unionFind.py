#! python3

#
# Implementation of the UnionFind data structure described here:
# 		https://en.wikipedia.org/wiki/Disjoint-set_data_structure
#
# @2017 Florin Tulba (florintulba@yahoo.com)
#


class UfItem:
	''' Data specific for each item of the Union Find '''
	def __init__(self, id):
		self.ancestor = id 	# the most distant known ancestor
		self.rank = 0		# initial depth of a subtree rooted on this item (some children might migrate closer to the root of this item)

class UF:
	''' The UnionFind class '''
	def __init__(self, itemsCount):
		self.items = [UfItem(i) for i in range(itemsCount)]	# the managed items
		self.groups = itemsCount							# current count of the groups formed from the items

	def __repr__(self):
		mapping = {}
		for i in range(len(self.items)):
			parentId = self.parentOf(i)
			if parentId in mapping:
				mapping[parentId].append(i)
			else:
				mapping[parentId] = [i]

		result = '{} groups: {}'.format(self.groups, mapping)
		return result

	def parentOf(self, id):
		''' Find parent operation '''
		parentId = self.items[id].ancestor
		while id != parentId:
			self.items[id].ancestor = self.items[parentId].ancestor
			id = self.items[id].ancestor
			parentId = self.items[id].ancestor
		return id

	def join(self, id1, id2):
		''' Connect id1 & id2 '''
		id1, id2 = self.parentOf(id1), self.parentOf(id2)
		if id1 == id2:
			return

		rank1, rank2 = self.items[id1].rank, self.items[id2].rank
		if rank1 < rank2:
			self.items[id1].ancestor = id2
		else:
			self.items[id2].ancestor = id1

		if rank1 == rank2:
			self.items[id1].rank += 1

		self.groups -= 1

if __name__ == "__main__":
	uf = UF(10)
	print("Initial uf:")
	print(uf)
#	uf.join(2, 1)
#	uf.join(0, 4)
#	uf.join(0, 5)
#	uf.join(6, 3)
#	uf.join(7, 4)
#	uf.join(9, 4)
#	uf.join(2, 6)
#	uf.join(2, 5)
#	uf.join(1, 8)
#	print(uf)

	uf.join(0, 3)
	print(uf)
	uf.join(4, 5)
	print(uf)
	uf.join(1, 9)
	print(uf)
	uf.join(2, 8)
	print(uf)
	uf.join(7, 4)
	print(uf)
	uf.join(9, 0)
	print(uf)
	uf.join(7, 8)
	print(uf)
	uf.join(1, 6)
	print(uf)
	uf.join(0, 5)
	print(uf)
