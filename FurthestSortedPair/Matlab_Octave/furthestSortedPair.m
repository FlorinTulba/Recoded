%{
    Part of the FurthestSortedPair project,
    that determines the most distant pair of sorted elements within an array

    @2017 Florin Tulba (florintulba@yahoo.com)
%}

function [ left right comparesCount ] = furthestSortedPair( vals )
%{
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
%}

    left = 1;
    right = 1;
    comparesCount = 0;
    valuesCount = length(vals);
    if valuesCount < 2
        return % There's no pair of elements yet
    end

	left = 1;
    leftVal = vals(left);
    right = valuesCount;
    rightVal = vals(right);
	comparesCount = comparesCount + 1;
	if leftVal < rightVal
        return
    end

	maxDist = 0;
    bestLeft = left;
    bestRight = left;

	% First inspection of the array considering first element as the left pair member
    
	rightOptions = zeros(valuesCount - 1, 3); % 3 columns: maxVal, idxFromRight, coverage
    rightOptions(1, : ) = [rightVal, right, 0]; % let coverage field be set to 1 within the loop
    ptrRightOptions = 1; % last relevant row in rightOptions
    
    while right > left + maxDist
    	comparesCount = comparesCount + 1;
		if rightVal > rightOptions(ptrRightOptions, 1)
			% new maximum, which can be then compared against leftVal
           	comparesCount = comparesCount + 1;
			if leftVal < rightVal
				maxDist = right - left;
                bestRight = right;
				break
            end

			% add the new max only if this matters when left=2
            ptrRightOptions = ptrRightOptions + 1;
			rightOptions(ptrRightOptions, : ) = [rightVal, right, 1];

        else
            % just increment coverage
            rightOptions(ptrRightOptions, 3) = ...
                rightOptions(ptrRightOptions, 3) + 1;
        end
        right = right - 1;
        rightVal = vals(right);
    end
    
	% Checking all remaining potential left pair members
	minVal = leftVal; % init min
    left = 2;
    while left + maxDist < valuesCount
		% a value should be popped out for each iteration
        if rightOptions(ptrRightOptions, 3) > 1
            rightOptions(ptrRightOptions, 3) = ...
                rightOptions(ptrRightOptions, 3) - 1;
        else
            ptrRightOptions = ptrRightOptions - 1;
        end

		% assessing only local minimum left pair members
    	comparesCount = comparesCount + 1;
		if leftVal >= minVal
            left = left + 1;
            leftVal = vals(left);
			continue
        end

		minVal = leftVal; % renew min

		% Check if there is no right pair member
		assert(ptrRightOptions > 0);
    	comparesCount = comparesCount + 1;
		if leftVal >= rightOptions(ptrRightOptions, 1)
            left = left + 1;
            leftVal = vals(left);
			continue
        end

		% The appropriate right pair member can be found using binary search
		comparesCount = comparesCount + ceil(log2(ptrRightOptions)); % overestimate of the compare ops performed
        idxOfLargerVal = upperBound(rightOptions(1 : ptrRightOptions, 1), leftVal);

		% Discarding (rightOptions(idxOfLargerVal, 2) - left) - maxDist
		% elements from  rightOptions (if there are so many) and assigning 
        % rightOptions(idxOfLargerVal, 2) - left to maxDist
        bestLeft = left;
        bestRight = rightOptions(idxOfLargerVal, 2);
		newMaxDist = bestRight - bestLeft;
        toReduce = newMaxDist - maxDist;
        
        while toReduce > 0 && ptrRightOptions > 0
            avail = rightOptions(ptrRightOptions, 3);
            removing = min(avail, toReduce);
            if avail == removing
                ptrRightOptions = ptrRightOptions - 1;
            else
                rightOptions(ptrRightOptions, 3) = ...
                    rightOptions(ptrRightOptions, 3) - removing;
            end
            toReduce = toReduce - removing;
        end
		
        maxDist = newMaxDist;

        left = left + 1;
        leftVal = vals(left);
    end

	% At this point, rightOptions should be empty, or contain one option
	% with coverage 1
	assert(ptrRightOptions == 0 || ...
		(ptrRightOptions == 1 && rightOptions(1, 3) == 1));

    left = bestLeft;
    right = bestRight;
