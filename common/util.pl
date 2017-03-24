%
% Various utilities shared by several 'Recoded' projects in Prolog
%
% @2017 Florin Tulba (florintulba@yahoo.com)
%


% Fails for lines from the file that are empty or comments
relevantLine(LineStr) :-
	\+ LineStr = "",						% Ignore empty lines
	\+ string_code(1, LineStr, 35).			% Ignore lines containing comments (they start with '#', which has 35 as char code)


% Provides the trimmed version of the next non-empty and non-comment line from the file pointed by Fd.
% Fails on EOF.
skipUselessLines(Fd, Line) :-
    \+ at_end_of_stream(Fd),
    read_string(Fd, '\n', '\r\t ', _, LineStr),	% Trims the line from Spaces, Tabs and Carriage Returns
    ((relevantLine(LineStr), !, Line = LineStr) 
    	;
    skipUselessLines(Fd, Line)).

