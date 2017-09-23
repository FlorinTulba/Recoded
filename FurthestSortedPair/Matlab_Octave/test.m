%{
    Part of the FurthestSortedPair project,
    that determines the most distant pair of sorted elements within an array

    @2017 Florin Tulba (florintulba@yahoo.com)
%}

clear all; close all; clc

% Define a skeleton for the signal (array)
skeleton = [50 0 75 -30 100 -60 80 -80 150 -50 25 -65 -10 -70 -20 -45 -40];

% Interpolate each interval with a variable number of points depending on
% how large the difference between the hint points is
signal = skeleton(1);
skelLen = length(skeleton);
rangeSkel = max(skeleton) - min(skeleton);
for i=2:skelLen
    diff = skeleton(i) - skeleton(i-1);
    interpolatedPoints = max(3, 200 * (abs(diff) / rangeSkel) + randi([-5, 5]));
    interpolatedPoints = linspace(skeleton(i-1), skeleton(i), interpolatedPoints);
    signal = [signal interpolatedPoints(2:end)];
end

clear interpolatedPoints diff rangeSkel i skelLen

% Add some noise to the signal
sigLen = length(signal);
signal = signal + (rand(1, sigLen) - .5) * 20.;

[refLeft refRight] = referenceResult(signal);
[left right] = furthestSortedPair(signal);
assert(refLeft == left && refRight == right);

% Display the signal (array) and a red line for the furthest sorted pair
% found
plot(signal), grid on, axis tight
hold on
line([refLeft refRight],[signal(refLeft) signal(refRight)],'Color','r')
fprintf('The most distant pair of sorted elements is %d elements apart.\n', ...
    right-left)
