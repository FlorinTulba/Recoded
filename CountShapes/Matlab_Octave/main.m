%{
    Part of the CountShapes project.

    @2017 Florin Tulba (florintulba@yahoo.com)
%}

close all; clear all; clc

%% Loading a figure (containing a shape to be processed)

% Select a file from below and comment the rest
fileName = 'count6shapes.png';
%fileName = 'count9shapes.png';
%fileName = 'count100shapes.png';
%fileName = 'count673shapes.png';

I = imread(sprintf('../TestFigures/%s', fileName)); Ired = ~(I(:, :, 1)>200);

fig = figure; subplot(221), imshow(I), title('Initial Image')

%% Detecting preliminary lines

%{
Setting 89.5 below as upper limit produces redundant horizontal lines,
since the angles rewind from 89.999 to -90 (both mean horizontal lines).
The rewind also flips the sign of the corresponding 'rho' (radius) value.
So, when the shape contains horizontal lines, they will appear as
local high values on the left and right margins of the Hough diagram,
but on opposite sides of the horizontal axis (they are symmetrical
around the center of the diagram, not around its vertical axis).

Because perfectly horizontal lines are more likely to be present within the
shapes, the diagram keeps the value -90 (exactly horizontal) and
drops the less likely values 88 - 89.5 (slightly decreasing slopes).

Lines were considered almost parallel if the difference of their Theta
(slope complements) is less than 2 degrees. This explains the previously
mentioned 88 value (90-2).
%}
[H,T,R] = hough(Ired, 'RhoResolution',.5, 'Theta',-90:0.5:87.5);
log_1pH = log(1+H); maxLog_1pH = max(log_1pH(:));
subplot(222), imshow(maxLog_1pH - log_1pH, [], 'XData',T, 'YData',R, 'InitialMagnification','fit')
xlabel('\theta'), ylabel('r'), title('Hough Transform'), axis on, axis normal, hold on

% Setting here a value around 0.125*max(H(:)) as threshold might help
% finding more lines (when some are not detected with 0.25*max(H(:))
P = houghpeaks(H, 150, 'Threshold',0.25*max(H(:)), 'NHoodSize',[5 5]);
x = T(P(:,2)); y = R(P(:,1)); plot(x,y, 's', 'color','red'); clear x y

lines = houghlines(Ired, T, R, P, 'FillGap',50, 'MinLength',7);

clear T R P H log_1pH maxLog_1pH

%{
 The obtained lines might:
    a) be stretching outside the shape
    b) need extending to reach the shape's edges/vertices
    c) miss the intended intersection points
    d) contain duplicates - different reported segments laying on the same actual line

 Goals:
    - getting the correct number of lines
    - ensure the lines reach shape's edges/vertices
    - infer the correct number of intersections within the shape and
        assign them a name and place them on the corresponding lines

 Solution:
    1. Tuning Hough transform parameters to find the seeds of all lines,
        then discarding any duplicates
    2. Computing the intersections of all pairs of lines
    3. Discarding the intersections which fall outside the shape
    4. Clustering the remaining intersection points 
%}


%% Discarding duplicate lines

[SortedThetas, Indices] = sort([lines(:).theta]);
lines = lines(Indices);

if length(lines) < 3
    disp('Detected less than 3 lines! A shape needs at least 3 lines!')
    return
end

duplicates = [];
ConsecDiffs = SortedThetas(2:end) - SortedThetas(1:end-1); % all values are >= 0
% ConsecDiffs are mapped to lines(1 : end-1)!

Indices = find(ConsecDiffs < 2);
i = 1; lim = length(Indices);
while i <= lim
    idx = Indices(i);
    delta = idx - i;
    j = i + 1;
    while j <= lim
        if Indices(j) - j ~= delta
            break
        end
        j = j + 1;
    end
    % here Indices(i:j-1) are consecutive values
    
    delta = j - i;
    [SortedRhos, IndicesRho] = sort([lines(idx : idx+delta).rho]);
    lines(idx : idx+delta) = lines((idx-1) + IndicesRho);
    ConsecDiffs = SortedRhos(2:end) - SortedRhos(1:end-1); % all values are >= 0
    % ConsecDiffs are mapped to lines(idx : idx+delta-1)!

    IndicesRho = find(ConsecDiffs < 4); % max rho diff (4 length units) to still consider lines as duplicates
    if ~isempty(IndicesRho)
        % lines(idx + IndicesRho) are duplicates of their corresponding previous lines

        % Normally, the longest segment among a duplicates group is the correct
        % segment, as the shortest ones are able to satisfy slightly different slopes
        IndicesRho = idx + IndicesRho;
        k = 1; limK = length(IndicesRho);
        while k <= limK
            idx = IndicesRho(k);
            delta = idx - k;
            l = k + 1;
            while l <= limK
                if IndicesRho(l) - l ~= delta
                    break
                end
                l = l + 1;
            end
            % here IndicesRho(k:l-1) are consecutive values
            % Their corresponding lines are lines(idx-1 : idx+l-k-1)
            delta = l - k;
            SegmentLengths = zeros(delta+1,1);
            for m = 1 : (delta+1)
                lineIdx = idx + m - 2;
                SegmentLengths(m) = ...
                    norm(lines(lineIdx).point1 - lines(lineIdx).point2);
            end
            [~, maxLenPos] = max(SegmentLengths(:));
            duplicates = [duplicates, ((idx-1) + [0:(maxLenPos-2), maxLenPos:delta])];

            k = l;
        end
    end
    
    i = j;
end

lines(duplicates) = [];

clear duplicates SortedThetas SortedRhos Indices IndicesRho ConsecDiffs
clear lim limK i j k l m idx delta SegmentLengths maxLenPos lineIdx

%% Display filtered lines

subplot(223), imshow(ones(size(Ired))), xlabel('Final Candidate Lines'), hold on
for k = 1 : length(lines)
    xy = [lines(k).point1; lines(k).point2];
    plot(xy(:,1), xy(:,2), 'LineWidth',1, 'Color','black');
    
    plot(xy(1,1), xy(1,2), 'x', 'LineWidth',1, 'Color','red');
    plot(xy(2,1), xy(2,2), 'x', 'LineWidth',1, 'Color','red');
end

clear x y xy k

%% Determine the exterior of the shape (to be able to discard exterior line intersections)

%{
 Determining the exterior of the shape:

 The drawn figures contain some segments which:
    - don't close the shape
    - are too short to land on their target edge
    - are too long and step outside the shape
    - should start from a vertex of the shape, but they fail to do so

 Steps:
	1. The exterior edges of the shape need to close the shape, but the segments
        responsible for that might break this prerequisite. A morphological
        closing would solve this issue, but dilation is simply enough and also
        more resistant to the flood filling step from below (avoids overflow fills)
	2. Since the shape is drawn inside the figure, point (1,1) always falls
        outside the shape. Therefore, a flood fill operation starting from (1,1)
        will flip the points from the exterior of the shape
    3. The difference between the flood filled image and the dilation of
        the initial one should represent the exterior of the shape
%}
se = strel('square', 5);
Idilated = imdilate(Ired, se);
Idilated(1,1) = false; % ensure the dilation keeps point (1,1) black

% 8-neighborhood might overflow fill when there are thin diagonal exterior shape segments!
Ifilled = imfill(Idilated, [1 1], 4);
ExtShape = logical(Ifilled - Idilated);

clear Idilated Ifilled se

%% Obtain the line intersections

[imRows imCols] = size(Ired);
linesCount = length(lines);

% uniquePoints matrix contains:
% - the pair of coordinates for each unique point
% - the count of intersections represented by each point
% The size is foundPoints x 3. Each row looks like: [x y count]
uniquePoints = [];
foundPoints = 0;

intersPoints = NaN(linesCount, linesCount);
% The 2 coordinates represent: indexOfFirstLine, indexOfSecondLine
% NaN values mean the lines are parallel or they coincide (on main diagonal only)
% Otherwise, the stored value represents a pointer towards a column in uniquePoints

% The loop from below performs also a simple form of clustering.
% 'clusterdata' might be used, as well:
%       https://uk.mathworks.com/help/stats/clusterdata.html
for i = 1 : (linesCount-1)
    li = lines(i);
    for j = (i+1) : linesCount
        lj = lines(j);
        [x y] = intersectionPoint(li.rho, li.theta, lj.rho, lj.theta);
        %fprintf('l%d/\\l%d = [%f, %f]\n', i,j, x,y)
        rx = round(x); ry = round(y);
        if isnan(rx) || isnan(ry) || ...
                rx < 1 || ry < 1 || ...
                rx > imCols || ry > imRows || ...
                ExtShape(ry, rx)
            continue
        end
        
        %{
            Compare p against all unique points found so far.
            If found a nearby point, compute the new centroid.
            Otherwise, just append the new point to uniquePoints.
        %}
        if foundPoints > 0
            diffX = abs(uniquePoints(:, 1) - x);
            smallDiffX = find(diffX < 6); % x offset threshold = 6; return all such values
            diffY = abs(uniquePoints(smallDiffX, 2) - y); % check only rows with small diffX
            smallDiffY = find(diffY < 6, 1); % y offset threshold = 6; find at most one value
            if ~isempty(smallDiffY)
                Pos = smallDiffX(smallDiffY(1));
                prevCount = uniquePoints(Pos, 3);
                newCount = prevCount + 1;
                uniquePoints(Pos,:) = [...
                    (prevCount * uniquePoints(Pos, 1:2) + [x y]) / newCount, ...
                	newCount];
            else
                foundPoints = foundPoints + 1;
                uniquePoints = [uniquePoints ; [x y 1]];
                Pos = foundPoints;
            end
        else
            uniquePoints = [x y 1]; % 1 means just one intersection yet
            foundPoints = 1;
            Pos = 1;
        end
        
        intersPoints(i, j) = Pos; intersPoints(j, i) = Pos;
    end
end

% Naming the points: A..Z, then A1..Z1, then A2..Z2 a.s.o.
pointNames = cell(foundPoints, 1);
loops = floor(foundPoints / 26);
rest = mod(foundPoints, 26);
if foundPoints <= 26
    for i = 1 : foundPoints
        pointNames{i} = char('A'-1 + i);
    end
else
    for i = 1 : 26
        pointNames{i} = char('A'-1 + i);
    end
    x = 26;
    for j = 1 : (loops-1)
        for i = 1 : 26
            pointNames{x+i} = ...
                sprintf('%s%i', char('A'-1 + i), j);
        end
        x = x + 26;
    end
    for i = 1 : rest
        pointNames{x+i} = ...
                sprintf('%s%i', char('A'-1 + i), loops);
    end
end

% Creating a cell array for the lines - each line with a vector of pointers
% towards the intersection points touched by that line and also sorted,
% first by x coordinate, then by y coordinate.
lines = cell(linesCount, 1);
for i = 1 : linesCount
    li = unique(intersPoints(~isnan(intersPoints(:, i)), i));
    [Sorted Indices] = sort(uniquePoints(li, 1));
    if Sorted(end) - Sorted(1) < 2
        % When x coordinates are almost identical, sort by y coordinates
        [~, Indices] = sort(uniquePoints(li, 2));
    end
    lines{i} = li(Indices);
end

clear i j li lj x y rx ry Pos prevCount newCount loops rest ExtShape
clear smallDiffX smallDiffY diffX diffY Sorted Indices imRows imCols

%% Display final lines and the names of the intersections

subplot(224), imshow(ones(size(Ired))), xlabel('Labeled Intersections '), hold on
for i = 1 : linesCount
    segmentEnds = lines{i}([1, end]);
    coords = uniquePoints(segmentEnds, 1:2);
    plot(coords(:,1), coords(:,2), 'LineWidth',1, 'Color','black');
end

for i = 1 : foundPoints
    pos = uniquePoints(i, 1:2);
    text(pos(1)-3, pos(2), pointNames{i}, 'Color','red', 'FontSize',15)
end

hold off

% Expanding the subplots:
% See https://uk.mathworks.com/matlabcentral/newsreader/view_thread/149202
% concerning the used function from below
subplotsqueeze(fig, 1.25)

clear i segmentEnds coords pos fig

%% Construct the ShapeCounter and save its configuration 
fd = fopen(sprintf('../TestFigures/TextVersions/%s.txt', fileName), 'wt');
fprintf(fd, '# File generated by the Matlab/Octave version\n# based on the image "../%s"\n',...
    fileName);
fprintf(fd, '# The intersection points are likely to be labeled differently compared to the original image.\n');
fprintf(fd, '# Each of the following rows displays all the points from a separate full line from the image.\n');
config = cell(1, linesCount);
for i = 1 : linesCount
    points = lines{i};
    pointsCount = length(points);
    configI = cell(1, pointsCount);
    for j = 1 : pointsCount
        configI{j} = pointNames{points(j)};
        fprintf(fd, '%s ', configI{j});
    end
    config{i} = configI;
    fprintf(fd, '\n');
end

fclose(fd);

sc = ShapeCounter(config);

clear i j points pointsCount configI fd

%% Launch the ShapeCounter
sc.process();
totalShapes = sc.triangles + sc.convexQuadrilaterals;
fprintf('There are %d triangles and %d convex quadrilaterals, which means %d convex shapes in total.\n', ...
    sc.triangles, sc.convexQuadrilaterals, totalShapes);

clear totalShapes