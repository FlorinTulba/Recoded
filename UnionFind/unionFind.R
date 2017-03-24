#
# Implementation of the UnionFind data structure described here:
# 		https://en.wikipedia.org/wiki/Disjoint-set_data_structure
#
# @2017 Florin Tulba (florintulba@yahoo.com)
#


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
    # the decrements ensure same output as in programming languages with 0-based indexing
    cat(sprintf('%3d - %3d :', id1-1L, id2-1L))
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
	cat(sprintf("%3d", length(unique(mapping[1,]))), 'groups:')
	i <- 1; parentId <- mapping['ParentId', i]
	repeat {
	    # the decrements ensure same output as in programming languages with 0-based indexing
		cat(parentId-1L, '{', mapping['Id', i]-1L, '')
		repeat {
			i <- i + 1
			if(i > n)
				break
			newParentId = mapping['ParentId', i]
			if(parentId != newParentId) {
				parentId <- newParentId
				break
			}
			# the decrements ensure same output as in programming languages with 0-based indexing
			cat(mapping['Id', i]-1L, '')
		}
		cat('}')
		if(i > n)
			break
		cat(', ')
	}
}

conn <- file('testScenario.txt', 'r')

source('../common/util.R') # for loading 'nextRelevantLine'
if(length(line <- nextRelevantLine(conn)) > 0) {
    n <- as.integer(line)			# items count
    if(n < 0L) {
        cat("Items count should be >= 0!\n")
    } else {
        if(n < 2L)
            cat('Note that this problem makes sense only for at least 2 elements!\n')

        ids <- 1L:n 					# items' ids
        ancestors <- ids				# initial items are also their own root ancestors

        # The rank of an item is the depth of a subtree rooted on this item (some children might migrate closer to the root of this item)
        ranks <- rep(0L, n)	# initial items' ranks are all 0

        # data frame with columns ancestor, rank
        uf <- data.frame(Ancestor=ancestors, Rank=ranks)
        cat(' Initially:'); cat(strUf(uf), '\n')

        noIssues <- TRUE
        while(length(line <- nextRelevantLine(conn)) > 0) {
            tokens <- strsplit(line, ' ', fixed = TRUE)[[1]]
            if(length(tokens) != 2) {
                cat(sprintf('Line "%s" contains less/more than 2 items!\n', line))
                noIssues <- FALSE
                break
            }

            idx1 <- as.integer(tokens[1])
            idx2 <- as.integer(tokens[2])
            if(idx1 < 0 || idx2 < 0 || idx1 >= n || idx2 >= n) {
                cat(sprintf('Line "%s" contains invalid indices!\n', line))
                noIssues <- FALSE
                break
            }

            # the increments ensure 1-based indexing as required in R
            join(uf, idx1+1L, idx2+1L); cat(strUf(uf), '\n')
        }
        if(noIssues)
            cat('No other pairs to connect in this scenario!')
    }
} else {
    cat("Couldn't read the items count!\n")
}

close(conn)
