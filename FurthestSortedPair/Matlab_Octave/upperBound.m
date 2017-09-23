%{
    Part of the FurthestSortedPair project,
    that determines the most distant pair of sorted elements within an array

    @2017 Florin Tulba (florintulba@yahoo.com)
%}

function idx = upperBound( vals, x )
%UPPERBOUND
% returns the index of first y > x from vals or [] when x >= max(vals)
%
% vals is a sorted array in ascending order
    if isempty(vals) || x >= vals(end)
        idx = [];
        return
    end
    
    len = length(vals);
    if len == 1
        idx = 1;
        return
    end
    
    mid = ceil(len / 2);
    if vals(mid) > x
        % keep mid, as it might be the first larger element than x
        idx = upperBound(vals(1:mid), x);
    else
        assert(mid + 1 <= len);
        idx = mid + upperBound(vals(mid+1 : len), x);
    end
