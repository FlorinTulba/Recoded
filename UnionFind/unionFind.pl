%
% Implementation of the UnionFind data structure described here:
% 		https://en.wikipedia.org/wiki/Disjoint-set_data_structure
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

% parentOf(Ancestors, Id, UpdatedAncestors, ParentId)
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

% join(Id1, Id2, PrevAncestors, PrevRanks, NextAncestors, NextRanks)
join(Id1, Id2, PrevAncestors, PrevRanks, NextAncestors, NextRanks) :-
	parentOf(PrevAncestors, Id1, PrevAncestors_, ParentId1),
	parentOf(PrevAncestors_, Id2, NextAncestors_, ParentId2),
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

% show(Ancestors)
show(Ancestors) :-
	findall(membership(Id, ParentId), parentOf(Ancestors, Id, _, ParentId), Membership),
	setof([ParentId, Members], setof(Id, member(membership(Id, ParentId), Membership), Members), Mapping),
	length(Mapping, Groups),
	writef('%d groups: %w\n', [Groups, Mapping]).

% initUF(N, Ancestors, Ranks)
initUF(0, [], []) :-
	!.
initUF(N, Ancestors, [0 | NextRanks]) :-
	N_1 is N - 1,
	initUF(N_1, NextAncestors, NextRanks),
	append(NextAncestors, [N_1], Ancestors).

checkUF :-
	N is 10,
	initUF(N, Ancestors0, Ranks0),
	writeln('Initial uf:'),
	show(Ancestors0),
	join(0, 3, Ancestors0, Ranks0, Ancestors1, Ranks1),
	show(Ancestors1),
	join(4, 5, Ancestors1, Ranks1, Ancestors2, Ranks2),
	show(Ancestors2),
	join(1, 9, Ancestors2, Ranks2, Ancestors3, Ranks3),
	show(Ancestors3),
	join(2, 8, Ancestors3, Ranks3, Ancestors4, Ranks4),
	show(Ancestors4),
	join(7, 4, Ancestors4, Ranks4, Ancestors5, Ranks5),
	show(Ancestors5),
	join(9, 0, Ancestors5, Ranks5, Ancestors6, Ranks6),
	show(Ancestors6),
	join(7, 8, Ancestors6, Ranks6, Ancestors7, Ranks7),
	show(Ancestors7),
	join(1, 6, Ancestors7, Ranks7, Ancestors8, Ranks8),
	show(Ancestors8),
	join(0, 5, Ancestors8, Ranks8, Ancestors9, _),
	show(Ancestors9),
	!.
