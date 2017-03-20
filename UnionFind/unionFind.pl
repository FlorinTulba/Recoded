%
% Implementation of the UnionFind data structure described here:
% 		https://en.wikipedia.org/wiki/Disjoint-set_data_structure
%
% Tested with SWI-Prolog 7.2.3
%
% @2017 Florin Tulba (florintulba@yahoo.com)
%


% listPrefix(List, N, Prefix) - returns first N items from List
listPrefix(_, N, []) :-
	N =< 0,
	!.	
listPrefix(List, N, Prefix) :-
	length(List, Len),
    (Len =< N ->
    	Prefix = List
    		;
    	(
	    	length(Prefix, N),
	        append(Prefix, _, List)
    	)
    ).

% listSuffix(List, N, Suffix) - returns the items from List starting from index N (0-based)
listSuffix(List, N, List) :-
	N =< 0,
	!.	
listSuffix(List, N, Suffix) :-
	length(List, Len),
    (Len =< N ->
    	Suffix = []
    		;
    	(
	    	SuffixLen is Len - N,
	    	length(Suffix, SuffixLen),
	        append(_, Suffix, List)
    	)
	).

% Returns the parent (ParentId) of Id, based on the known Ancestors. It updates these ancestors.
parentOf(Ancestors, Id, UpdatedAncestors, ParentId) :-
	nth0(Id, Ancestors, ParentId1),
	(Id =:= ParentId1 ->
		(ParentId = Id, UpdatedAncestors = Ancestors)
			;
		(
			nth0(ParentId1, Ancestors, ParentId2),
			Idp1 is Id + 1,
			listPrefix(Ancestors, Id, Prefix),
			listSuffix(Ancestors, Idp1, Suffix),
			append(Prefix, [ParentId2 | Suffix], Ancestors1),
			parentOf(Ancestors1, ParentId2, UpdatedAncestors, ParentId)
		)
	).

% Unites Id1 & Id2 based on PrevAncestors & PrevRanks. The updated ancestors and ranks are returned.
join(Id1, Id2, PrevAncestors, PrevRanks, NextAncestors, NextRanks) :-
	writef('%3r - %3r : ', [Id1, Id2]),
	(parentOf(PrevAncestors, Id1, PrevAncestors_, ParentId1) -> true ;
		(writef('The parent of %d was not found!\n', [Id1]), fail)),
	(parentOf(PrevAncestors_, Id2, NextAncestors_, ParentId2) -> true ;
		(writef('The parent of %d was not found!\n', [Id2]), fail)),
	(
		(
			ParentId1 =:= ParentId2,
			!,
			NextRanks = PrevRanks,
			NextAncestors = NextAncestors_
		)
			;
		(
			nth0(ParentId1, PrevRanks, Rank1),
			nth0(ParentId2, PrevRanks, Rank2),
			(Rank1 < Rank2 ->
				(
					ParentId1p1 is ParentId1 + 1,
					listPrefix(NextAncestors_, ParentId1, Prefix),
					listSuffix(NextAncestors_, ParentId1p1, Suffix),
					append(Prefix, [ParentId2 | Suffix], NextAncestors)
				)
					;
				(
					ParentId2p1 is ParentId2 + 1,
					listPrefix(NextAncestors_, ParentId2, Prefix),
					listSuffix(NextAncestors_, ParentId2p1, Suffix),
					append(Prefix, [ParentId1 | Suffix], NextAncestors)
				)
			),

			(Rank1 =:= Rank2 ->
				(
					Rank1p1 is Rank1 + 1,
					ParentId1p1 is ParentId1 + 1,
					listPrefix(PrevRanks, ParentId1, PrefixRanks),
					listSuffix(PrevRanks, ParentId1p1, SuffixRanks),
					append(PrefixRanks, [Rank1p1 | SuffixRanks], NextRanks)
				)
					;
				NextRanks = PrevRanks
			)
		)
	).

% Displays the content of the UnionFind data structure - the ancestors together with their descendants
show(Ancestors) :-
	findall(membership(Id, ParentId), parentOf(Ancestors, Id, _, ParentId), Membership),
	setof([ParentId, Members], setof(Id, member(membership(Id, ParentId), Membership), Members), Mapping),
	length(Mapping, Groups),
	writef('%3r groups: %w\n', [Groups, Mapping]).

% initUF(N, Ancestors, Ranks) - initializes the UnionFind data structure with the elements count (N).
% Returns the initial Ancestors & Ranks.
initUF(0, [], []) :-
	!.
initUF(N, Ancestors, [0 | NextRanks]) :-
	N_1 is N - 1,
	initUF(N_1, NextAncestors, NextRanks),
	append(NextAncestors, [N_1], Ancestors).



% Fails for lines from the test scenario file that are empty or comments
relevantLine(LineStr) :-
	\+ LineStr = "",						% Ignore empty lines
	\+ string_code(1, LineStr, 35).			% Ignore lines containing comments (they start with '#', which has 35 as char code)

% Provides the trimmed version of the next non-empty and non-comment line from the test scenario file.
% Fails on EOF.
skipUselessLines(Fd, Line) :-
    \+ at_end_of_stream(Fd),
    read_string(Fd, '\n', '\r\t ', _, LineStr),	% Trims the line from Spaces, Tabs and Carriage Returns
    ((relevantLine(LineStr), !, Line = LineStr) 
    	;
    skipUselessLines(Fd, Line)).

% Reads the items count (N) from the test scenario file
readItemsCount(Fd, N) :-
	skipUselessLines(Fd, Line),
    split_string(Line, ' \t', '', [_]),		% expect a single token on the line
	number_string(N, Line),
	(N < 2 -> writeln('Note that this problem makes sense only for at least 2 elements!') ; true).

% Executes all join operations requested in the test scenario file
processUnions(Fd, Ancestors, Ranks) :-
	skipUselessLines(Fd, Line) ->
	(
		split_string(Line, ' \t', '', Tokens),
		(Tokens = [Idx1str, Idx2str] ->
			(
				(number_string(Idx1, Idx1str), integer(Idx1),
				number_string(Idx2, Idx2str), integer(Idx2)) ->
					(
						join(Idx1, Idx2, Ancestors, Ranks, Ancestors1, Ranks1), show(Ancestors1),
						processUnions(Fd, Ancestors1, Ranks1)
					)
						;
					(
						writef('Line "%w" contains non-integer value(s)!\n', [Line]),
						fail
					)
			)
				;
			(
				writef('Line "%w" contains less/more than 2 items!\n', [Line]),
				fail
			)
		)
	)
		;
	writeln('No other pairs to connect in this scenario!').



% Predicate for launching the program
main :-
	open('testScenario.txt', read, Fd),
	(readItemsCount(Fd, N) ->
		(
			ReadItemsCount = true,
			initUF(N, Ancestors, Ranks), write(' Initially: '), show(Ancestors),
			(processUnions(Fd, Ancestors, Ranks) ->
				ProcessedUnions = true
					;
				ProcessedUnions = fail
			)
		)
			;
		(
			ReadItemsCount = fail,
			writeln("Couldn't read the items count!")
		)
	),
	close(Fd),
	ReadItemsCount,
	ProcessedUnions.
