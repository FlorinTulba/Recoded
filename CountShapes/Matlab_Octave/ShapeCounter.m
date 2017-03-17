%{
    Part of the CountShapes project.

    @2017 Florin Tulba (florintulba@yahoo.com)
%}

classdef ShapeCounter < handle
    %SHAPECOUNTER Counts triangles and convex quadrilaterals from a figure
    
    properties
		N = 0 						% Points Count
		L               			% Lines Count
		triangles = 0 				% Count of triangles
		convexQuadrilaterals = 0 	% Count of convex quadrilaterals
		pointNames = {}             % the names of the points
		lineMembers = []			% the indices of the members of each line (list of bitarray-s)
		connections = []			% points connectivity matrix (list of bitarray-s)
		membership = []             % membership of points to the lines (list of bitarray-s)
		membershipAsRanks = {}		% for each point a map of pairs: lineIdx and rankWithinLine
    end
    
    methods
        % Configures the ShapeCounter based on the sequences of named points from the lines from the figure.
		% Prepares the entire infrastructure needed while counting the shapes.
        function self = ShapeCounter(lines)
            if nargin > 0
                self.L = length(lines);
                pointsIndices = containers.Map();
                for lineIdx = 1 : self.L
                    line = lines{lineIdx};
                    pointsOnLine = length(line);
                    memberIndices = zeros(pointsOnLine, 1);
                    memberIndicesBitset = bitarray(self.N);
                    for pointRank = 1 : pointsOnLine
                        pointName = line{pointRank};
                        if ~pointsIndices.isKey(pointName)
                            self.N = self.N + 1;
                            self.pointNames{self.N} = pointName;
                            pointIdx = self.N;
                            pointsIndices(pointName) = pointIdx;
                            for prevMemIdx = 1 : length(self.lineMembers)
                                self.lineMembers(prevMemIdx).prepend(0);
                            end
                            for prevConnIdx = 1 : length(self.connections)
                                self.connections(prevConnIdx).prepend(0);
                            end
                            self.connections = [self.connections ; bitarray(self.N)];
                            memberIndicesBitset.prepend(1);
                            self.membership = [self.membership ; bitarray(self.L)];
                            self.membershipAsRanks{self.N} = ...
                                remove(containers.Map(0,0), 0);
                        else
                            pointIdx = pointsIndices(pointName);
                            memberIndicesBitset.set(pointIdx);
                        end
                        lastPointConns = self.connections(pointIdx);
                        for prevMemIdx = 1 : (pointRank - 1)
                            lastPointConns.set(memberIndices(prevMemIdx));
                            self.connections(memberIndices(prevMemIdx)).set(pointIdx);
                        end
                        memberIndices(pointRank) = pointIdx;
                        self.membership(pointIdx).set(lineIdx);
                        self.membershipAsRanks{pointIdx}(lineIdx) = pointRank;
                    end
                    self.lineMembers = [self.lineMembers ; memberIndicesBitset];
                end

                %{
                % Uncomment if interested in inspecting the correctness of the configuration based on the current <lines> parameter
                for i = 1 : self.N
                    fprintf('%s: connections %s ; member of lines %s ; pos in lines {', ...
                        self.pointNames{i}, self.connections(i).str(true), self.membership(i).str(true));
                    aux = self.membershipAsRanks{i};
                    theKeys = keys(aux); theValues = values(aux);
                    for lineIdx = 1 : length(aux)
                        fprintf('%d(l%d) ', theValues{lineIdx}-1, theKeys{lineIdx}-1);
                    end
                    fprintf('}\n');
                end
                for i = 1 : self.L
                    fprintf('L%d: members %s\n', i-1, self.lineMembers(i).str(true));
                end
                %}
            else
                self.L = 0;
            end
        end
        
        % Performs the actual shape counting
        function process(self)
            % One step for ensuring the uniqueness of the solutions:
            % a mask to prevent the shapes found later from using points before P1.
            maskP1 = bitarray(self.N); maskP1.compl();
            for p1 = 1 : (self.N - 2)
                mem1 = self.membership(p1);
                maskP1.set(p1, 0); % Ignore connections before and including P1
                connOfP1Bitset = self.connections(p1).and(maskP1);
                connOfP1 = connOfP1Bitset.indicesOf1();
                countConnOfP1 = connOfP1Bitset.countedOnes;
                if countConnOfP1 < 2
                    continue % Triangles require 2 connected points to P1. If they are not available, check next available P1
                end

                for idxP2 = 1 : (countConnOfP1-1)
                    p2 = connOfP1(idxP2);
                    mem2 = self.membership(p2);
                    mem1and2 = mem1.and(mem2);
                    for idxLastP = (idxP2 + 1) : countConnOfP1
                        lastP = connOfP1(idxLastP);
                        memLast = self.membership(lastP);
                        if mem1and2.and(memLast).countedOnes > 0 % coll(p1, p2, lastP)
                            continue	% Ignore collinear points
                        end

                        if self.connections(p2).get(lastP)
                            self.triangles = self.triangles + 1;
                            fprintf('<%s%s%s> ', ...
                                self.pointNames{p1}, self.pointNames{p2}, self.pointNames{lastP});
                        end

                        connOfP2_LastP_Bitset = self.connections(p2).and(...
                            self.connections(lastP).and(maskP1));
                        mem1and2or2andLast = mem1and2.or(mem2.and(memLast));
                        possibleP3 = connOfP2_LastP_Bitset.indicesOf1();
                        for idxP3 = 1 : connOfP2_LastP_Bitset.countedOnes
                            p3 = possibleP3(idxP3);
                            mem3 = self.membership(p3);
                            if mem1and2or2andLast.and(mem3).countedOnes > 0 % coll(p1, p2, p3) || coll(lastP, p2, p3)
                                continue	% Ignore collinear points
                            end

                            if self.convex(p1, mem1, p2, mem2, p3, mem3, lastP, memLast)
                                self.convexQuadrilaterals = self.convexQuadrilaterals + 1;
                                fprintf('[%s%s%s%s] ', ...
                                    self.pointNames{p1}, self.pointNames{p2}, ...
                                    self.pointNames{p3}, self.pointNames{lastP});
                            end
                        end
                    end
                end
            end
            fprintf('\n');
        end
        
        % Checks convexity of p1-p4 quadrilateral, based on the membership
        % of each point to the available lines
        function tf = convex(self, p1, mem1, p2, mem2, p3, mem3, p4, mem4)
            if nargin < 9
                tf = false;
                return;
            end
            
            % Extended p1-p2 and p3-p4 shouldn't touch
            if ~self.allowedIntersection([p1, p2], [p3, p4], mem1.and(mem2), mem3, mem4)
                tf = false;
                return;
            end

            % Extended p2-p3 and p4-p1 shouldn't touch
            if ~self.allowedIntersection([p2, p3], [p4, p1], mem2.and(mem3), mem4, mem1)
                tf = false;
                return;
            end

            tf = true;
        end
        
        %{
		Returns true only if the extended lines l1 and l2 don't intersect, or intersect strictly outside the shape described by the 4 points from l1 and l2.
		Parameters:
			l1 		- one line from a potential quadrilateral
			l2 		- the line across from l1 in the potential quadrilateral
			memL1 	- 'and'-ed memberships (which lines include each point) of the 2 points from l1
			memL2_1 - membership (which lines include the point) of one point from l2
			memL2_1 - membership (which lines include the point) of the other point from l2
        %}
        function tf = allowedIntersection(self, l1, l2, memL1, memL2_1, memL2_2)
            if nargin < 6
                tf = false;
                return;
            end
            
            lineIdxPair1 = memL1.indicesOf1();
            if memL2_1.get(lineIdxPair1) || memL2_2.get(lineIdxPair1)
                tf = false; % one of the provided points from L2 are inside L1
                return
            end

            lineIdxPair2 = memL2_1.and(memL2_2).indicesOf1();
            intersectionPoint = self.lineMembers(lineIdxPair1).and(...
                self.lineMembers(lineIdxPair2)).indicesOf1();
            if intersectionPoint
                % The found intersection point should fall outside the segment l1
                % The check relies on the fact that lines specify the contained points in order
                rank1 = self.membershipAsRanks{l1(1)}(lineIdxPair1);
                rank2 = self.membershipAsRanks{l1(2)}(lineIdxPair1);
                if rank1 > rank2
                    aux = rank1; rank1 = rank2; rank2 = aux;
                end
                intersectionPointMembership = self.membershipAsRanks{intersectionPoint};
                rank = intersectionPointMembership(lineIdxPair1);
                if rank1 <= rank && rank <= rank2
                    tf = false;
                    return
                end

                % The found intersection point should fall outside the segment l2
                rank1 = self.membershipAsRanks{l2(1)}(lineIdxPair2);
                rank2 = self.membershipAsRanks{l2(2)}(lineIdxPair2);
                if rank1 > rank2
                    aux = rank1; rank1 = rank2; rank2 = aux;
                end
                rank = intersectionPointMembership(lineIdxPair2);
                if rank1 <= rank && rank <= rank2
                    tf = false;
                    return
                end
            end
            tf = true;
        end
    end    
end

