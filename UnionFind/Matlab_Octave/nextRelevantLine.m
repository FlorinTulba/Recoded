%{
    Part of the implementation of the UnionFind data structure described here:
        https://en.wikipedia.org/wiki/Disjoint-set_data_structure

    @2017 Florin Tulba (florintulba@yahoo.com)
%}


function line = nextRelevantLine(fd)
%NEXTRELEVANTLINE Returns the next non-empty and non-comment line from file
% 'testScenario.txt' represented by parameter 'fd'.
% The comments from the scenario file start with '#'
    line = fgetl(fd);
    while ischar(line)
        if ~isempty(line) && line(1) ~= '#'
            return
        end
        line = fgetl(fd);
    end
end
