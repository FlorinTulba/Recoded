%{
    Part of the implementation of the UnionFind data structure described here:
        https://en.wikipedia.org/wiki/Disjoint-set_data_structure

    @2017 Florin Tulba (florintulba@yahoo.com)
%}


% The UnionFind class
classdef Uf < handle
    properties
        items = []  % the managed items
        groups = 0  % current count of the groups formed from the items
    end
    
    methods
        % Create itemCount items that are initially separated
        function uf = Uf(itemsCount)
            if nargin > 0
                uf.groups = itemsCount;
                Items(1, itemsCount) = UfItem;
                uf.items = Items;
                for i=1:itemsCount
                    uf.items(i) = UfItem(i);
                end
            end
            fprintf(' Initially:%s\n', uf.str());
        end
        
        % Find parent operation
        function parentId = parentOf(uf, id)
            if id < 1 || id > length(uf.items)
                disp('Invalid index!')
                parentId = [];
                return
            end
            parentId = uf.items(id).ancestor;
            while id ~= parentId
                uf.items(id).ancestor = uf.items(parentId).ancestor;
                id = uf.items(id).ancestor;
                parentId = uf.items(id).ancestor;
            end
            parentId = id;
        end
        
        % Connect id1 & id2
        function join(uf, id1, id2)
            fprintf('%3d - %3d :', id1-1, id2-1); % display 0-based indices
            id1 = uf.parentOf(id1); id2 = uf.parentOf(id2);
            if id1 == id2
                fprintf('%s\n', uf.str());
                return
            end
            
            rank1 = uf.items(id1).rank; rank2 = uf.items(id2).rank;
            if rank1 < rank2
                uf.items(id1).ancestor = id2;
            else
                uf.items(id2).ancestor = id1;
            end
            
            if rank1 == rank2
                uf.items(id1).rank = uf.items(id1).rank + 1;
            end
            
            uf.groups = uf.groups - 1;
            
            fprintf('%s\n', uf.str());
        end
        
        % 'to string' method
        function Str = str(uf)
            if ~isempty(uf.items)
                mapping = containers.Map(uf.parentOf(1), 1, 'uniformvalues', false);
                for i=2:length(uf.items)
                    parentId = uf.parentOf(i);
                    if mapping.isKey(parentId)
                        mapping(parentId) = [mapping(parentId); i];
                    else
                        mapping(parentId) = i;
                    end
                end
                Str = sprintf('%3d groups: ', uf.groups);
                keys = mapping.keys; vals = mapping.values;
                for i=1:uf.groups
                    Str = sprintf('%s%d {', Str, keys{i}-1); % display 0-based indices
                    members = vals{i};
                    for j=1:length(members)
                        Str = sprintf('%s%d ', Str, members(j)-1); % display 0-based indices
                    end
                    Str = sprintf('%s}  ', Str);
                end
            else
                Str = '';
            end
        end
    end    
end
