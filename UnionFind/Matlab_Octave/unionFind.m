%{
    Part of the implementation of the UnionFind data structure described here:
        https://en.wikipedia.org/wiki/Disjoint-set_data_structure

    @2017 Florin Tulba (florintulba@yahoo.com)
%}

clear all; clc

fd = fopen('../testScenario.txt');

line = nextRelevantLine(fd);
if ischar(line)
    n = sscanf(line, '%d');
    if n < 0
        disp('Items count must be >= 0!')
    else
        uf = Uf(n);
        if n < 2
            disp('Note that this problem makes sense only for at least 2 elements!')
        end
        line = nextRelevantLine(fd);
        while ischar(line)
            indices = sscanf(line, '%d%d');
            idx1 = indices(1); idx2 = indices(2);
            uf.join(idx1+1, idx2+1); % Matlab needs 1-based indexing
            line = nextRelevantLine(fd);
        end
    end
else
    disp('Could not read the items count!')
end

fclose(fd);
