#
# Determine the most distant pair of sorted elements within a given array
#
# Required packages:
#   <combinat> - https://cran.r-project.org/web/packages/combinat/index.html
#
# @2017 Florin Tulba (florintulba@yahoo.com)
#

library(combinat) # for generating permutations with 'permn'


# Slow, but safe method with O(N^2) for finding the furthest pair of sorted values
# within vals.
#
# It uses a main loop to consider all possible left members of the pair.
# There is also an inner loop that checks for corresponding right members of the
# pair starting from the right end of the array and stopping when the pair spacing
# is better / inferior than the previous best.
referenceResult <- function(vals) {
    valsCount <- length(vals)
    if(valsCount == 0)
        return(0)

    maxDist <- 0
    left <- 1
    while(left + maxDist < valsCount) {
        leftVal <- vals[left]
        right <- valsCount
        while(right > left + maxDist) {
            if(leftVal < vals[right]) {
                maxDist <- right - left
                break
            }
            right <- right - 1
        }
        left <- left + 1
    }

    maxDist
}

#' O(log(N)) binary-search like function
#' @return the index of first x from sortedSeq (sorted in ascending order),
#' so that x > val. If there is no such x, it returns NA
upperBound <- function(sortedSeq, val) {
    len <- length(sortedSeq)
    if(len == 0L || sortedSeq[len] <= val)
        return(NA)

    if(len == 1L)
        return(1)

    mid <- as.integer(ceiling(len/2))
    if(sortedSeq[mid] > val)
        # keep checking mid, as it might be exactly the 1st element > val
        return(upperBound(sortedSeq[1 : mid], val))

    mid + upperBound(sortedSeq[(mid+1) : len], val)
}

#' This approach of finding the furthest pair of sorted values within vals works
#' in O(N*log(N)).
#'
#' @param vals the array of elements
#' @param comparesCountOut overestimated value of the number of required compare operations
#'
#' @return the distance between the furthest pair of sorted values of the array
#'
#' It also uses a main loop to consider all possible left members of the pair.
#' However it considers the following facts:
#'
#'     - larger left members of the pair than previously considered cannot deliver
#' better result. This means skipping left pair members larger than the
#' minimum left pair member previously analyzed
#'
#' - similarly, for a given left pair member, the corresponding right pair member
#' might be strictly one of the (updated) maximum values found while traversing
#' the array from right towards left
#'
#' - the analysis of the right pair member for the first possible left pair member
#' (the very first loop) allows improving the next passes. So, the index of each
#' newly encountered maximum value (right to left traversal) can be recorded.
#'
#' - the search for the first larger value than left pair member within the
#' previously mentioned array of stored maximum values (recorded in ascending
#' order) can be performed using a binary search, obtaining a log(N) for the
#' inner loop
improvedMethod <- function(vals, comparesCountOut) {
    valuesCount <- length(vals)
    comparesCount <- 0
    if(valuesCount < 2) {
        eval.parent(substitute(comparesCountOut <- comparesCount))
        return(0) # There's no pair of elements yet
    }

    left <- 1
    leftVal <- vals[1]
    right <- valuesCount
    rightVal <- vals[right]
    comparesCount <- comparesCount + 1
    if(leftVal < rightVal) {
        eval.parent(substitute(comparesCountOut <- comparesCount))
        return(valuesCount-1)
    }

    maxDist <- 0

    # First inspection of the array considering first element as the left pair member
    rightOptions <- data.frame(
        MaxValues = rightVal,
        Indices = right,
        Coverages = 0) # let coverage field be set to 1 within the loop
    ptrArrays <- 1 # maintains the current position within previous lists
    while(right > maxDist + left) {
        # new maximum, which can be then compared against leftVal
        comparesCount <- comparesCount + 1
        if(rightVal > rightOptions$MaxValues[nrow(rightOptions)]) {
            comparesCount <- comparesCount + 1
            if(leftVal < rightVal) {
                maxDist <- right - left
                break
            }

            # add the new max only if this matters when left=2
            rightOptions <- rbind(rightOptions,
                                  data.frame(
                                      MaxValues = rightVal,
                                      Indices = right,
                                      Coverages = 1))
            ptrArrays <- ptrArrays + 1

        } else {
            rightOptions$Coverages[nrow(rightOptions)] <-
                rightOptions$Coverages[nrow(rightOptions)] + 1
        }

        right <- right - 1
        rightVal <- vals[right]
    }

    # Checking all remaining potential left pair members
    minVal = leftVal; # init min
    left = 2
    while(left + maxDist < valuesCount) {
        leftVal <- vals[left]

        # a value should be popped out for each iteration
        if(rightOptions$Coverages[ptrArrays] > 1) {
            rightOptions$Coverages[ptrArrays] <-
                rightOptions$Coverages[ptrArrays] - 1;
        } else {
            ptrArrays <- ptrArrays - 1
        }

        # assessing only local minimum left pair members
        comparesCount <- comparesCount + 1
        if(leftVal >= minVal) {
            left <- left + 1
            next
        }

        minVal <- leftVal # renew min

        # Check if there is no right pair member
        stopifnot(ptrArrays > 0)
        comparesCount <- comparesCount + 1
        if(leftVal >= rightOptions$MaxValues[ptrArrays]) {
            left <- left + 1
            next
        }

        # The appropriate right pair member can be found using binary search
        comparesCount <- comparesCount +
            ceiling(log2(ptrArrays)) # overestimate of the compare ops performed
        idxOfLargerVal <- upperBound(
            rightOptions$MaxValues[1 : ptrArrays],
            leftVal);

        # Discarding (rightOptions$Indices[idxOfLargerVal] - left) - maxDist elements
        # from rightOptions (if there are so many)
        # and assigning rightOptions$Indices[idxOfLargerVal] - left to maxDist
        newMaxDist <- rightOptions$Indices[idxOfLargerVal] - left;

        toRemove <- newMaxDist - maxDist
        while(toRemove > 0 && ptrArrays > 0) {
            avail <- rightOptions$Coverages[ptrArrays]
            removing <- min(avail, toRemove)
            if(avail == removing) {
                ptrArrays <- ptrArrays - 1
            } else {
                rightOptions$Coverages[ptrArrays] <-
                    rightOptions$Coverages[ptrArrays] - removing
            }
            toRemove <- toRemove - removing
        }

        maxDist <- newMaxDist

        left <- left + 1
    }

    # At this point, rightOptions should be empty,
    # or contain one option with coverage 1
    stopifnot(ptrArrays == 0 || (ptrArrays == 1 && rightOptions$Coverages[1] == 1))

    eval.parent(substitute(comparesCountOut <- comparesCount))
    maxDist
}

# Compares the results of the 2 approaches and reports errors
# and compare operations count
checkUseCase <- function(vals, errorsCount, verbose = FALSE) {
    comparesCount <- 0
    errors <- errorsCount

    refRes <- referenceResult(vals)
    res <- improvedMethod(vals, comparesCount)
    if(refRes != res) {
        errors <- errors + 1
        cat("For the array below, the expected result was", refRes,
              ", but obtained", res, "instead.\n", file=stderr())
        cat(vals, '\n', file=stderr())

    } else if(verbose) {
        cat("Furthest sorted pair of elements is at a distance of", res,
              "in the array from below. It needed", comparesCount,
              "compare ops.\n")
        cat(vals, '\n')
    }
    eval.parent(substitute(errorsCount <- errors))

    comparesCount
}

TIMES <- 1000			# count of random arrays to be checked
VALUES_COUNT <- 1000		# size of each random array

errorsCount <- 0

vals <- c() # the array containing the values for the tests
# Empty array case
checkUseCase(vals, errorsCount)

# Single element array case
vals <- 100
checkUseCase(vals, errorsCount)

# Sorted array case
vals <- 0 : (VALUES_COUNT-1)
checkUseCase(vals, errorsCount, TRUE)

# Descending sorted array case
vals <- (VALUES_COUNT-1) : 0
checkUseCase(vals, errorsCount, TRUE)

# Random arrays cases
cat("Checking random arrays ...\n")
vals <- sample(vals)
checkUseCase(vals, errorsCount, TRUE)

for(t in 1 : TIMES) {
    vals <- sample(vals)
    checkUseCase(vals, errorsCount)
}

# Searching for the worst case scenario within the 40320 permutations
# of an array of 8 elements
cat("Looking for the worst case scenario ...\n")
maxComparesCount <- 0
worstConfig <- c()
vals <- 0 : 7
for(perm in permn(vals)) {
    comparesCount <- checkUseCase(perm, errorsCount)
    if(comparesCount > maxComparesCount) {
        maxComparesCount <- comparesCount
        worstConfig <- perm
    }
}

cat("The worst configuration from below produced", maxComparesCount,
      "compare ops.\n")
cat(worstConfig, "\n")

# Inspecting the worst case scenario found above:
vals <- c(7, 6, 5, 4, 3, 2, 1, 0)
comparesCount <- 0
improvedMethod(vals, comparesCount)
cat("The worst case scenario required", comparesCount, "compare ops.\n")

if(errorsCount > 0) {
    cat("There were", errorsCount, "errors!\n", file=stderr())
} else {
    cat("There were no errors\n")
}

