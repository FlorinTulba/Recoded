%{
Helper script used for generating the 'seed' of figure '../TestFigures/count25651shapes.png'.
First, it recreates '../TestFigures/count673shapes.png'. Then it replicates this 4 times,
2 times horizontally and 2 times vertically.

This is a 'seed' figure, as some of the lines don't reach the edges of the larger figure.
However, the implemented line interpreter from 'main.m' is able to detect these 'seed' segments
and extend them towards the external edges of the larger figure.
%}

clear all; close all; clc

% Recreating '../TestFigures/count673shapes.png' (more precisely-drawn and without the labels)
sz = uint32(251); mid = uint32((sz+1)/2); sz2 = sz*sz;
I = true(sz);

% Drawing the secondary bisecting lines:
alpha = pi/8; tanAlpha = tan(alpha);
I(round(1 : (double(sz) + tanAlpha) : double(sz2))) = false; % Draw secondary bisecting line
I = I & I'; I = I & fliplr(I); I = I & flipud(I); % Produce all symmetrical secondary bisecting lines

I(1,:) = false; I(sz,:) = false; I(:,1) = false; I(:,sz) = false; % Draw borders
I(mid,:) = false; I(:,mid) = false; % Draw horizontal & vertical lines through center
I(1 : (sz+1) : sz2) = false; % Draw main diagonal
I(sz : (sz-1) : sz2) = false;  % Draw secondary diagonal

% The larger figure without padding ('I' replicated on both axes while sharing the common edge)
I4 = true(2*sz-1);
I4(1:sz, 1:sz) = I; I4(sz:end, 1:sz) = I; I4(1:sz, sz:end) = I; I4(sz:end, sz:end) = I;

% The larger figure with 10 pixels padding
paddedI4 = true(2*sz-1+20);
paddedI4(11:2*sz+9, 11:2*sz+9) = I4;

imshow(paddedI4)