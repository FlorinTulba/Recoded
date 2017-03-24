/*
	Various utilities shared by several 'Recoded' projects in C++

	@2017 Florin Tulba (florintulba@yahoo.com)
*/

#ifndef H_UTIL
#define H_UTIL

#include <string>
#include <istream>

/// Returns a copy of the parameter 's' with space-like characters removed from both ends
std::string trim(const std::string &s);

/**
Discards all following empty / comment lines from the stream 'is'
until reaching EOF or a non-empty & non-comment line,
whose trimmed version is assigned to the 'line' parameter.

Comment lines start with '#'.

@return the updated 'is', in order to let the method be used like
		while(nextRelevantLine(is, line)) {...}
*/
std::istream& nextRelevantLine(std::istream &is, std::string &line);

#endif // H_UTIL