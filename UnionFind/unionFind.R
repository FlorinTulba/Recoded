#
# Implementation of the UnionFind data structure described here:
# 		https://en.wikipedia.org/wiki/Disjoint-set_data_structure
#
# @2017 Florin Tulba (florintulba@yahoo.com)
#


n <- 10							# items count
ids <- 1:n 						# items' ids
ancestors <- ids				# initial items are also their own root ancestors

# The rank of an item is the depth of a subtree rooted on this item (some children might migrate closer to the root of this item)
ranks <- rep(as.integer(0), n)	# initial items' ranks are all 0

# data frame with columns ancestor, rank
uf <- data.frame(Ancestor=ancestors, Rank=ranks)

# Find parent operation
parentOf <- function(uf, id) {
	ufTemp <- uf
	parentId <- ufTemp[id, 'Ancestor']
	while(id != parentId) {
		ufTemp[id, 'Ancestor'] <- ufTemp[parentId, 'Ancestor']
		id <- ufTemp[id, 'Ancestor']
		parentId <- ufTemp[id, 'Ancestor']
	}
	eval.parent(substitute(uf <- ufTemp))
	as.integer(id)
}

# Connect id1 & id2
join <- function(uf, id1, id2) {
	ufTemp <- uf
	id1 <- parentOf(ufTemp, id1)
	id2 <- parentOf(ufTemp, id2)
	if(id1 == id2) {
		eval.parent(substitute(uf <- ufTemp))
		return
	}

	rank1 <- ufTemp[id1, 'Rank']
	rank2 <- ufTemp[id2, 'Rank']
	if(rank1 < rank2)
		ufTemp[id1, 'Ancestor'] <- id2
	else
		ufTemp[id2, 'Ancestor'] <- id1

	if(rank1 == rank2)
		ufTemp[id1, 'Rank'] <- ufTemp[id1, 'Rank'] + as.integer(1)

	eval.parent(substitute(uf <- ufTemp))
}

# Displays the UnionFind content without mutating it (mutator method 'parentOf' will produce a local copy of uf)
strUf <- function(uf) {
	mapping <- sapply(ids, function(id) c(ParentId=parentOf(uf, id), Id=id))
	mapping <- mapping[,order(mapping['ParentId',])]
	cat(length(unique(mapping[1,])), 'groups:')
	i <- 1; parentId <- mapping['ParentId', i]
	repeat {
		cat(parentId, '{', mapping['Id', i], '')
		repeat {
			i <- i + 1
			if(i > n)
				break
			newParentId = mapping['ParentId', i]
			if(parentId != newParentId) {
				parentId <- newParentId
				break
			}
			cat(mapping['Id', i], '')
		}
		cat('}')
		if(i > n)
			break
		cat(', ')
	}
}

cat('Initial uf:\n')
cat(strUf(uf), '\n')

join(uf, 1, 4)
cat(strUf(uf), '\n')
join(uf, 5, 6)
cat(strUf(uf), '\n')
join(uf, 2, 10)
cat(strUf(uf), '\n')
join(uf, 3, 9)
cat(strUf(uf), '\n')
join(uf, 8, 5)
cat(strUf(uf), '\n')
join(uf, 10, 1)
cat(strUf(uf), '\n')
join(uf, 8, 9)
cat(strUf(uf), '\n')
join(uf, 2, 7)
cat(strUf(uf), '\n')
join(uf, 1, 6)
cat(strUf(uf), '\n')
