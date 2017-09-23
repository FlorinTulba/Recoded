%{
    Part of the FurthestSortedPair project,
    that determines the most distant pair of sorted elements within an array

    @2017 Florin Tulba (florintulba@yahoo.com)
%}

function [left right] = referenceResult( vals )
    %{
    Slow, but safe method with O(N^2) for finding the furthest pair of sorted values
    within vals.

    It uses a main loop to consider all possible left members of the pair.
    There is also an inner loop that checks for corresponding right members of the
    pair starting from the right end of the array and stopping when the pair spacing
    is better / inferior than the previous best.
    %}
    left = 0;
    right = 0;
    if isempty(vals)
        return
    end
    
    valsCount = length(vals);
	maxDist = 0;

    bestLeft = 0;
    bestRight = 0;
    left = 1;
    while left + maxDist < valsCount
        leftVal = vals(left);
        right = valsCount;
        while right > left + maxDist
            if leftVal < vals(right)
				maxDist = right - left;
                bestLeft = left;
                bestRight = right;
				break
            end
            right = right - 1;
        end
        left = left + 1;
    end
    left = bestLeft;
    right = bestRight;
