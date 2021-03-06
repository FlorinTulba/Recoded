#
# Counting all possible triangles and (convex) quadrilaterals
# from geometric figures traversed by a number of lines.
#
# See the tested figures in 'TestFigures/'.
#
# Required packages:
#   <bit> - https://cran.r-project.org/web/packages/bit/index.html
#
# @2017 Florin Tulba (florintulba@yahoo.com)
#


library(bit) # for bit vectors
source('../common/util.R') # for loading 'nextRelevantLine'


# Select one of the following scenario files (uncommenting it and commenting the others)
# or create a new scenario and open your own file:

# Scenario from figure 'count6Shapes.png'
#conn <- file('TestFigures/TextVersions/count6shapes.png.txt', 'r') # Generated by the figure interpreter written in Matlab/Octave
conn <- file('TestFigures/TextVersions/count6shapes.txt', 'r') # Manually generated (same labels as in the corresponding figure)

# Scenario from figure 'count9Shapes.png'
#conn <- file('TestFigures/TextVersions/count9shapes.png.txt', 'r') # Generated by the figure interpreter written in Matlab/Octave
#conn <- file('TestFigures/TextVersions/count9shapes.txt', 'r') # Manually generated (same labels as in the corresponding figure)

# Scenario from figure 'count100Shapes.png'
#conn <- file('TestFigures/TextVersions/count100shapes.png.txt', 'r') # Generated by the figure interpreter written in Matlab/Octave
#conn <- file('TestFigures/TextVersions/count100shapes.txt', 'r') # Manually generated (same labels as in the corresponding figure)

# Scenario from figure 'count673Shapes.png'
#conn <- file('TestFigures/TextVersions/count673shapes.png.txt', 'r') # Generated by the figure interpreter written in Matlab/Octave
#conn <- file('TestFigures/TextVersions/count673shapes.txt', 'r') # Manually generated (same labels as in the corresponding figure)

# Scenario from figure 'count25651Shapes.png'
#conn <- file('TestFigures/TextVersions/count25651shapes.png.txt', 'r') # Generated by the figure interpreter written in Matlab/Octave

noIssues <- T
lines <- list()
linesCount <- 0
while(length(line <- nextRelevantLine(conn)) > 0) {
    linePoints <- strsplit(line, ' ', fixed = T)[[1]]
    if(length(linePoints) < 2) {
        cat(sprintf('This "%s" is not a correct line, since it contains less than 2 points!\n', line))
        noIssues <- F
        break
    }
    linesCount <- linesCount + 1
    lines[[linesCount]] <- linePoints
}
close(conn)

stopifnot(noIssues)


# a ShapeCounter object
sc <- list(
    N = 0L,						# Points Count
    L = 0L,						# Lines Count
    triangles = 0L,				# Count of triangles
    convexQuadrilaterals = 0L, 	# Count of convex quadrilaterals
    pointNames = character(),   # the names of the points
    lineMembers = list(),		# the indices of the members of each line (list of bitarray-s)
    connections = list(),       # points connectivity matrix (list of bitarray-s)
    membership = list(),		# membership of points to the lines (list of bitarray-s)
    membershipAsRanks = list()  # for each point a data.frame between lineIdx and rankWithinLine
    )

# Initialization of the ShapeCounter object using the points specified on each line
sc$L <- length(lines)
for(lineIdx in seq_len(sc$L)) {
    line <- lines[[lineIdx]]
    pointsOnLine <- length(line)
    memberIndices <- integer()
    memberIndicesBitset <- bit(sc$N)
    for(pointRank in seq_len(pointsOnLine)) {
        pointName <- line[pointRank]
        pointIdx <- sc$N + 1L # assume point name is not yet in pointNames
        for(pIdx in seq_len(sc$N)) {
            if(pointName == sc$pointNames[pIdx]) {
                pointIdx <- pIdx
                break
            }
        }
        if(pointIdx > sc$N) { # point name was not found in pointNames
            sc$pointNames <- append(sc$pointNames, pointName)
            sc$N <- sc$N + 1L
            for(prevMemIdx in seq_len(length(sc$lineMembers)))
                sc$lineMembers[[prevMemIdx]] <- c.bit(sc$lineMembers[[prevMemIdx]], 0)
            for(prevConnIdx in seq_len(length(sc$connections)))
                sc$connections[[prevConnIdx]] <- c.bit(sc$connections[[prevConnIdx]], 0)
            sc$connections[[1L + length(sc$connections)]] <- bit(sc$N)
            memberIndicesBitset <- c.bit(memberIndicesBitset, 1)
            sc$membership[[1L + length(sc$membership)]] <- bit(sc$L)
            sc$membershipAsRanks[[1L + length(sc$membershipAsRanks)]] <-
                data.frame(LineIdx=integer(), RankWithinLine=integer())
        } else {
            memberIndicesBitset[[pointIdx]] <- 1
        }
        for(prevMemIdx in seq_len(length(memberIndices))) {
            sc$connections[[pointIdx]][[memberIndices[prevMemIdx]]] <- 1
            sc$connections[[memberIndices[prevMemIdx]]][[pointIdx]] <- 1
        }
        memberIndices <- append(memberIndices, pointIdx)
        sc$membership[[pointIdx]][[lineIdx]] <- 1
        sc$membershipAsRanks[[pointIdx]] <-
            rbind(sc$membershipAsRanks[[pointIdx]],
                  data.frame(LineIdx=lineIdx, RankWithinLine=pointRank))
    }
    sc$lineMembers[[1L + length(sc$lineMembers)]] <- memberIndicesBitset
}

# Uncomment if interested in inspecting the correctness of the configuration based on the current <lines> parameter
# for(i in seq_len(sc$N)) {
# 	cat(sc$pointNames[i], ': connections {', as.integer(sc$connections[[i]]),
# 	    '} ; member of lines {', as.integer(sc$membership[[i]]),'} ; pos in lines {', sep = '')
#     df <- sc$membershipAsRanks[[i]]
#     for(lineIdx in seq_len(dim(df)[1])) {
#         dfi <- df[lineIdx,]
#         cat(dfi$RankWithinLine - 1, '(l', dfi$LineIdx - 1, ') ', sep = '')
#     }
#     cat('}\n')
# }
# for(i in seq_len(sc$L))
# 	cat('L', i-1, ': members {', as.integer(sc$lineMembers[[i]]),'}\n', sep = '')



# Returns true only if the extended lines l1 and l2 don't intersect,
# or intersect strictly outside the shape described by the 4 points from l1 and l2.
#
# Parameters:
#     l1 	    - one line from a potential quadrilateral
#     l2 		- the line across from l1 in the potential quadrilateral
#     memL1 	- 'and'-ed memberships (which lines include each point) of the 2 points from l1
#     memL2_1   - membership (which lines include the point) of one point from l2
#     memL2_1   - membership (which lines include the point) of the other point from l2
allowedIntersection <- function(sc, l1, l2, memL1, memL2_1, memL2_2) {
    for(lineIdxPair1 in seq_len(sc$N))
        if(memL1[[lineIdxPair1]])
            break

    if(memL2_1[[lineIdxPair1]] || memL2_2[[lineIdxPair1]])
        return(FALSE) # one of the provided points from L2 are inside L1

    memL2 <- memL2_1 & memL2_2
    for(lineIdxPair2 in seq_len(sc$N))
        if(memL2[[lineIdxPair2]])
            break

    intersectionBitSet <- sc$lineMembers[[lineIdxPair1]] & sc$lineMembers[[lineIdxPair2]]
    foundIntersectionPoint <- FALSE
    for(intersectionPoint in seq_len(sc$N))
        if(intersectionBitSet[[intersectionPoint]]) {
            foundIntersectionPoint <- TRUE
            break
        }
    if(foundIntersectionPoint) {
        # The found intersection point should fall outside the segment l1
        # The check relies on the fact that lines specify the contained points in order
        aux <- sc$membershipAsRanks[[l1[1]]]
        rank1 <- aux[aux$LineIdx == lineIdxPair1,]$RankWithinLine
        aux <- sc$membershipAsRanks[[l1[2]]]
        rank2 <- aux[aux$LineIdx == lineIdxPair1,]$RankWithinLine
        if(rank1 > rank2) {
            aux <- rank1; rank1 <- rank2; rank2 <- aux
        }
        intersectionPointMembership <- sc$membershipAsRanks[[intersectionPoint]]
        rank <- intersectionPointMembership[
            intersectionPointMembership$LineIdx == lineIdxPair1,]$RankWithinLine
        if(rank1 <= rank && rank <= rank2)
            return(FALSE)

        # The found intersection point should fall outside the segment l2
        aux <- sc$membershipAsRanks[[l2[1]]]
        rank1 <- aux[aux$LineIdx == lineIdxPair2,]$RankWithinLine
        aux <- sc$membershipAsRanks[[l2[2]]]
        rank2 <- aux[aux$LineIdx == lineIdxPair2,]$RankWithinLine
        if(rank1 > rank2) {
            aux <- rank1; rank1 <- rank2; rank2 <- aux
        }
        rank <- intersectionPointMembership[
            intersectionPointMembership$LineIdx == lineIdxPair2,]$RankWithinLine
        if(rank1 <= rank && rank <= rank2)
            return(FALSE)
    }

    TRUE
}

# Checks convexity of p1-p4 quadrilateral, based on the membership of each point to the available lines
convex <- function(sc, p1, mem1, p2, mem2, p3, mem3, p4, mem4) {
    # Extended p1-p2 and p3-p4 shouldn't touch
    if( ! allowedIntersection(sc, c(p1, p2), c(p3, p4), (mem1 & mem2), mem3, mem4))
        return(FALSE)

    # Extended p2-p3 and p4-p1 shouldn't touch
    if( ! allowedIntersection(sc, c(p2, p3), c(p4, p1), (mem2 & mem3), mem4, mem1))
        return(FALSE)

    TRUE
}

# Performing the actual shape counting
# One step for ensuring the uniqueness of the solutions:
# a mask to prevent the shapes found later from using points before P1.
maskP1 <- ! bit(sc$N) # mask containing only 1
for(p1 in seq_len(sc$N - 2L)) {
    mem1 <- sc$membership[[p1]]
    maskP1[[p1]] <- 0 # Ignore connections before and including P1
    connOfP1Bitset <- sc$connections[[p1]] & maskP1
    countConnOfP1 <- sum(connOfP1Bitset)
    if(countConnOfP1 < 2)
        next # Triangles require 2 connected points to P1. If they are not available, check next available P1

    connOfP1 <- rep(0L, countConnOfP1)
    foundOnes <- 0L
    for(p in (p1 + 1L) : sc$N)
        if(connOfP1Bitset[[p]]) {
            foundOnes <- foundOnes + 1
            connOfP1[foundOnes] <- p
            if(foundOnes == countConnOfP1)
                break
        }
    for(idxP2 in seq_len(countConnOfP1 - 1L)) {
        p2 <- connOfP1[idxP2]
        mem2 <- sc$membership[[p2]]
        mem1and2 <- mem1 & mem2
        for(idxLastP in (idxP2 + 1L) : countConnOfP1) {
            lastP <- connOfP1[idxLastP]
            memLast <- sc$membership[[lastP]]
            if(any(mem1and2 & memLast)) # coll(p1, p2, lastP)
                next    # Ignore collinear points

            if(sc$connections[[p2]][[lastP]]) {
                sc$triangles <- sc$triangles + 1L
                cat('<', sc$pointNames[p1], sc$pointNames[p2], sc$pointNames[lastP], '> ', sep = '')
            }

            connOfP2_LastP_Bitset <- sc$connections[[p2]] & sc$connections[[lastP]] & maskP1
            mem1and2or2andLast <- mem1and2 | (mem2 & memLast)
            for(p3 in (p1 + 1L) : sc$N) {
                if( ! connOfP2_LastP_Bitset[[p3]])
                    next

                mem3 <- sc$membership[[p3]]
                if(any(mem1and2or2andLast & mem3)) # coll(p1, p2, p3) || coll(lastP, p2, p3)
                    next    # Ignore collinear points

                if(convex(sc, p1, mem1, p2, mem2, p3, mem3, lastP, memLast)) {
                    sc$convexQuadrilaterals <- sc$convexQuadrilaterals + 1L
                    cat('[', sc$pointNames[p1], sc$pointNames[p2], sc$pointNames[p3], sc$pointNames[lastP], '] ', sep = '')
                }
            }
        }
    }
}
cat('\n')

totalShapes <- sc$triangles + sc$convexQuadrilaterals
cat("There are", sc$triangles, "triangles and", sc$convexQuadrilaterals,
    "convex quadrilaterals, which means", totalShapes, "convex shapes in total.")
