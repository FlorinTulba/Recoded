%{
    Part of the implementation of the UnionFind data structure described here:
        https://en.wikipedia.org/wiki/Disjoint-set_data_structure

    @2017 Florin Tulba (florintulba@yahoo.com)
%}


% Data specific for each item of the Union Find
classdef UfItem
    properties
        ancestor    % the most distant known ancestor
        rank = 0    % initial depth of a subtree rooted on this item (some children might migrate closer to the root of this item)
    end
    
    methods
        function ufItem = UfItem(id)
            if  nargin > 0
                ufItem.ancestor = id;
            end
        end
    end    
end

