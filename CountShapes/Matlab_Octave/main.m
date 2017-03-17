%{
    Part of the CountShapes project.

    @2017 Florin Tulba (florintulba@yahoo.com)
%}

clear all; clc

% Scenario from figure 'count6Shapes.png'
sc = ShapeCounter({{'a', 'c'}, {'a', 'd'}, {'e', 'f'}, {'b', 'f', 'd', 'c'}, {'a', 'e', 'b'}});

% Scenario from figure 'count9Shapes.png'
%sc = ShapeCounter({{'a', 'c'}, {'b', 'd', 'f', 'c'}, {'a', 'e', 'b'}, {'e', 'g', 'f'}, {'a', 'g', 'd'}});

%{
% Scenario from figure 'count100Shapes.png'
sc = ShapeCounter({{'a', 'i'}, {'a', 'b', 'c', 'd', 'e'}, {'a', 'n', 'm', 'l', 'f'},...
    {'a', 'o', 'r', 'k', 'g'}, {'a', 'p', 'q', 'j', 'h'}, {'i', 'j', 'k', 'l', 'd'},...
    {'i', 'q', 'r', 'm', 'c'}, {'i', 'p', 'o', 'n', 'b'}, {'e', 'f', 'g', 'h', 'i'}});
%}

%{
% Scenario from figure 'count673Shapes.png'
sc = ShapeCounter({{'a', 'b', 'c', 'd', 'e'}, {'a', 's', 't', 'v', 'w', 'f'}, {'a', 'r', 'g1', 'z', 'i'},...
    {'a', 'q', 'f1', 'd1', 'c1', 'l'}, {'a', 'p', 'o', 'n', 'm'}, {'b', 's', 'r', 'f1', 'e1', 'm'},...
    {'c', 't', 'g1', 'b1', 'k'}, {'d', 'u', 'v', 'x', 'y', 'i'}, {'e', 'u', 't', 'r', 'q', 'p'},...
    {'e', 'v', 'g1', 'd1', 'm'}, {'e', 'w', 'x', 'z', 'a1', 'j'}, {'e', 'f', 'g', 'h', 'i'},...
    {'g', 'x', 'g1', 'f1', 'o'}, {'h', 'y', 'z', 'b1', 'c1', 'm'},...
    {'i', 'a1', 'b1', 'd1', 'e1', 'n'}, {'i', 'j', 'k', 'l', 'm'}});
%}

sc.process();
totalShapes = sc.triangles + sc.convexQuadrilaterals;
fprintf('There are %d triangles and %d convex quadrilaterals, which means %d convex shapes in total.\n', ...
    sc.triangles, sc.convexQuadrilaterals, totalShapes);
