%{

    Part of the implementation of the UnionFind data structure described here:
        https://en.wikipedia.org/wiki/Disjoint-set_data_structure

    @2017 Florin Tulba (florintulba@yahoo.com)
%}

clear all; clc

uf = Uf(10);
fprintf('Initial uf is:\n%s\n', uf.str());

uf.join(1, 4);
fprintf('%s\n', uf.str());
uf.join(5, 6);
fprintf('%s\n', uf.str());
uf.join(2, 10);
fprintf('%s\n', uf.str());
uf.join(3, 9);
fprintf('%s\n', uf.str());
uf.join(8, 5);
fprintf('%s\n', uf.str());
uf.join(10, 1);
fprintf('%s\n', uf.str());
uf.join(8, 9);
fprintf('%s\n', uf.str());
uf.join(2, 7);
fprintf('%s\n', uf.str());
uf.join(1, 6);
fprintf('%s\n', uf.str());
