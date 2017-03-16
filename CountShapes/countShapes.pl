%
% Counting all possible triangles and (convex) quadrilaterals from geometric figures traversed by a number of lines.
% See the tested figures in 'TestFigures/'.
%
% @2017 Florin Tulba (florintulba@yahoo.com)
%


%%%%%%%%%%%%%%%%%
% SCENARIO DATA %
%%%%%%%%%%%%%%%%%

% Every scenario comes with:
% - a list of points
% - a set of isolated lines which don't intersect any other segments passing within the figure.
%	The ends of these isolated edges are either corners or points of the external edges
% - a set of Collinear Points (at least 3 points) specified in the order they appear on a given line

%%% Scenario for figure 'count6Shapes.png'
points([a, b, c, d, e, f]).
isolatedLines([[a, c], [a, d], [e, f]]).
collPoints([[b, f, d, c], [a, e, b]]).

%%% Scenario for figure 'count9Shapes.png'
%points([a, b, c, d, e, f, g]).
%isolatedLines([[a, c]]).
%collPoints([[b, d, f, c], [a, e, b], [e, g, f], [a, g, d]]).

%%% Scenario for figure 'count100Shapes.png'
%points([a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r]).
%isolatedLines([[a, i]]).
%collPoints([[a, b, c, d, e], [a, n, m, l, f], [a, o, r, k, g], [a, p, q, j, h], [i, j, k, l, d], [i, q, r, m, c], [i, p, o, n, b], [e, f, g, h, i]]).

%%% Scenario for figure 'count673Shapes.png'
%points([a, b, c, d, e, f, g, h, i, j, k, l, m, n, o, p, q, r, s, t, u, v, w, x, y, z, a1, b1, c1, d1, e1, f1, g1]).
%isolatedLines([]).
%collPoints([[a,b,c,d,e], [a,s,t,v,w,f], [a,r,g1,z,i], [a,q,f1,d1,c1,l], [a,p,o,n,m], [b,s,r,f1,e1,m], [c,t,g1,b1,k], [d,u,v,x,y,i], [e,u,t,r,q,p], [e,v,g1,d1,m], [e,w,x,z,a1,j], [e,f,g,h,i], [g,x,g1,f1,o], [h,y,z,b1,c1,m], [i,a1,b1,d1,e1,n], [i,j,k,l,m]]).


%%%%%%%%%%%%%%
% PREDICATES %
%%%%%%%%%%%%%%

% Algorithm steps:
%
% Traverse Points:
% 	P1 - current point
% 	RestPoints - next points from Points
%
% All points before P1 are not allowed any more!
% 
% Let ConnP1AfterP1 be the points from RestPoints connected to P1
%
% Traverse ConnP1AfterP1:
% 	P2 - current point;
%	LastP - a different point connected to P1 beyond P2; LastP shouldn't be collinear with P1 & P2
% 	RestConnP1AfterP2 - next points from ConnP1AfterP1 (choices for LastP)
%	AllowedLastP - the points from RestConnP1AfterP2 connected to P1 and non-collinear with P1 & P2
%
% Traverse AllowedLastP:
% If P2 & LastP are connected, then increment triangles count.
%
% Let AllowedP3 be the points from the intersection of ConnP2AfterP1 and ConnLastPAfterP1 that generate valid convex quadrilaterals.
% Set convex quadrilaterals count as the length of AllowedP3.


% Check if 2 points form an isolated line
isIsolatedSegment(P1, P2) :-
	isolatedLines(IsolatedLines),
	(member([P1, P2], IsolatedLines), ! ; member([P2, P1], IsolatedLines)), !.

% Checks if 3 distinct points are collinear
coll(P1, P2, P3) :-
	P1 \= P2,
	P1 \= P3,
	P3 \= P2,
	collPoints(CollPoints),
	member(L, CollPoints),
	member(P1, L),
	member(P2, L),
	member(P3, L),
	!.

% Checks if 2 distinct points are connected
connected(P1, P2) :-
	P1 \= P2,
	(
		(
			collPoints(CollPoints),
			member(L, CollPoints),
			member(P1, L),
			member(P2, L), !
		)
			;
		isIsolatedSegment(P1, P2)
	),
	!.

% Ensures that the 2 lines intersect only outside the P1-P4 shape or don't intersect at all
extInters([P1, P1], _) :- fail, !.
extInters(_, [P3, P3]) :- fail, !.
extInters([P1, P2], [P3, P4]) :- 
	collPoints(CollPoints), isolatedLines(IsolatedLines),
	(member(L1, CollPoints) ; member(L1, IsolatedLines)),
	nth0(IdxP1, L1, P1), nth0(IdxP2, L1, P2),				% Find a line L1 that contains P1 & P2
	\+ (member(P3, L1), member(P4, L1)),					% But not P3, neither P4
	(member(L2, CollPoints) ; member(L2, IsolatedLines)),
	nth0(IdxP3, L2, P3), nth0(IdxP4, L2, P4),				% Find a line L2 that contains P3 & P4
	((member(P, L1), member(P, L2)) ->						% If there is a point P which belongs to both L1 & L2
		(
			nth0(IdxPL1, L1, P), nth0(IdxPL2, L2, P),		% Check that P falls outside both segments [P1, P2], [P3, P4]
			(IdxPL1-IdxP1)*(IdxPL1-IdxP2) > 0,
			(IdxPL2-IdxP3)*(IdxPL2-IdxP4) > 0
		)
			;
		true 												% The lines L1 & L2 don't intersect inside the figure or at all
	),
	!.

% Validates Triangles (3 non-collinear and connected points)
triangle(P1, P2, P3) :-
	P1 \= P2,
	P1 \= P3,
	P3 \= P2,
	\+ coll(P1, P2, P3),
	connected(P1, P2),
	connected(P1, P3),
	connected(P3, P2),
	!.

% Validates Convex Quadrilaterals - the order of the points should denote a perimeter traversal
% Convexity is enforced by ensuring that the extended opposing sides of the quadrilateral don't cross each other inside the quadrilateral.
convexQuadrilateral(P1, P2, P3, P4) :-
	P1 \= P2, P1 \= P3, P1 \= P4, P2 \= P3, P2 \= P4, P3 \= P4,
	connected(P1, P2), connected(P2, P3), \+ coll(P1, P2, P3),		% 4 non-collinear points, connected like P1-P2-P3-P4-P1
	connected(P3, P4), \+ coll(P2, P3, P4), connected(P4, P1), \+ coll(P3, P4, P1), % \+ coll(P4, P1, P2), - not necessary
	extInters([P1, P2], [P3, P4]), extInters([P2, P3], [P4, P1]),	% Ensure convexity and also that the segments are not diagonals
	!.

% Returns a list with all points connected to X, except first Skip points from the original Points list
% Contributes to the uniqueness of the solutions by avoiding previously analyzed points
connectionsOf(X, Skip, L) :-
	points(Points),
	findall(Y, (nth0(Idx, Points, Y), Idx > Skip, connected(X, Y)), L).
connectionsOf(X, L) :-
	points(Points),
	nth0(PosX, Points, X),
	connectionsOf(X, PosX, L).

% Sets LastP - the last point (from the triangle / convex quadrilateral) and optionally P3 for quadrilaterals
handleLastP(_, _, [], 0, 0) :- !.
handleLastP(P1, P2, [LastP | RestAllowedLastP], TriCount, QuadrCount) :-
	(connected(P2, LastP) ->						% if LastP is also connected to P2, P1-P2-LastP forms a triangle, since they are also non-collinear
		(
			writef('<%w%w%w> ', [P1, P2, LastP]),	% Comment it if not interested in the actually generated triangles
			%,triangle(P1, P2, LastP),				% Uncomment to check the triangle's validity
			TriCountThis is 1
		)
			;
		TriCountThis is 0
	),
	points(Points),
	nth0(IdxP1, Points, P1),
	connectionsOf(P2, IdxP1, ConnP2AfterP1),
	connectionsOf(LastP, IdxP1, ConnLastPAfterP1),	% Examine all P3 connected to P2 and LastP which are beyond P1 in the Points list (ensures solution uniqueness)
	findall(P3, (member(P3, ConnP2AfterP1), member(P3, ConnLastPAfterP1),
				\+ coll(P1, P2, P3), \+ coll(LastP, P2, P3),
				extInters([P1, P2], [P3, LastP]),
				extInters([P2, P3], [LastP, P1])
				,writef('[%w%w%w%w] ', [P1, P2, P3, LastP])		% Comment it if not interested in the actually generated convex quadrilaterals
				%,convexQuadrilateral(P1, P2, P3, LastP)		% Uncomment to check the convex quadrilateral's validity
				),
			AllowedP3),								% Collect all P3 that generate P1-P2-P3-LastP convex quadrilaterals
	length(AllowedP3, QuadrCountThis),
	handleLastP(P1, P2, RestAllowedLastP, TriCountRest, QuadrCountRest),	% Apply same algorithm for next LastP
	TriCount is TriCountRest + TriCountThis,
	QuadrCount is QuadrCountRest + QuadrCountThis.

% Chooses the 2nd point and last point (from the triangle / convex quadrilateral) as neighbors of P1
handleP2_lastP(_, [], 0, 0) :- !.
handleP2_lastP(P1, [P2 | RestConnP1AfterP2], TriCount, QuadrCount) :-
	findall(LastP, (member(LastP, RestConnP1AfterP2),
					\+ coll(P1, P2, LastP)),
			AllowedLastP),		% select LastP from the neighbors of P1 beyond P2 so that P1, P2, LastP are non-collinear
	handleLastP(P1, P2, AllowedLastP, TriCountThis, QuadrCountThis),
	handleP2_lastP(P1, RestConnP1AfterP2, TriCountRest, QuadrCountRest),	% Apply same algorithm for P2 taken from the next neighbors of P1
	TriCount is TriCountRest + TriCountThis,
	QuadrCount is QuadrCountRest + QuadrCountThis.

% Chooses the 1st point and further delegates the remaining tasks
handleP1([], 0, 0) :- !.
handleP1([P1 | RestPoints], TriCount, QuadrCount) :-
	findall(P, (member(P, RestPoints), connected(P1, P)), ConnP1AfterP1),
	handleP2_lastP(P1, ConnP1AfterP1, TriCountThis, QuadrCountThis),
	handleP1(RestPoints, TriCountRest, QuadrCountRest),	% Apply same algorithm for P1 taken from the remaining points
	TriCount is TriCountRest + TriCountThis,
	QuadrCount is QuadrCountRest + QuadrCountThis.

% Main predicate
countShapes :-
	points(Points),
	handleP1(Points, TriCount, QuadrCount),
	ShapesCount is TriCount + QuadrCount,
	writef('\n\nThere are %d triangles and %d convex quadrilaterals, which means %d convex shapes in total.\n', [TriCount, QuadrCount, ShapesCount]),
	!.
