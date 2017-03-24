/*
	Various utilities shared by several 'Recoded' projects in C++

	Compilable in g++ 5.4.0 with:
		g++ -std=c++14 -c -Ofast -Wall -o util.o util.cpp

	@2017 Florin Tulba (florintulba@yahoo.com)
*/

#include "util.h"

using namespace std;

string trim(const string &s) {
	auto itFront = cbegin(s);
	auto itBackRev = crbegin(s);
	for(auto itEnd = cend(s); itFront != itEnd && isspace(*itFront); ++itFront);
	for(auto itEnd = string::const_reverse_iterator(itFront);
		itBackRev != itEnd && isspace(*itBackRev); ++itBackRev);
	return string(itFront, itBackRev.base());
}

istream& nextRelevantLine(istream &is, string &line) {
	while(getline(is, line) || (line=string(), false)) { // Ensures empty string at EOF
		line = trim(line); // Gets read also of the '\r' returned on Unix-like systems
		if( ! line.empty() && line[0] != '#')
			break;
	}
	return is;
}
