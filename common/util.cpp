/*
	Various utilities shared by several 'Recoded' projects in C++

	Compilable in g++ 5.4.0 with:
		g++ -std=c++14 -c -Ofast -Wall -o util.o util.cpp

	@2017 Florin Tulba (florintulba@yahoo.com)
*/

#include "util.h"

#include <iostream>

using namespace std;
using namespace std::chrono;

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

Timer::Timer(const std::string &taskName_, size_t repetitions_/* = 1ULL*/) :
		taskName(taskName_),
		repetitions(repetitions_),
		// taskName & repetitions are initialized before starting the timer
		startedAt(high_resolution_clock::now()) {}

Timer::Timer(Timer &&t) :
		taskName(std::move(t.taskName)),
		repetitions(t.repetitions),
		startedAt(std::move(t.startedAt)) {
	t.repetitions = 0ULL; // sets t as invalid
}

Timer::~Timer() {
	const double elapsedS = elapsed(); // harmless call if the Timer isn't valid
	
	if(repetitions != 0ULL) // validity check after stopping the timer
		cout<<"Task '"<<taskName<<"' required: "<< elapsedS / (double)repetitions <<"s!"<<endl;
}

Timer& Timer::operator=(Timer &&t) {
	if(this != &t) {
		taskName = std::move(t.taskName);
		repetitions = t.repetitions;
		startedAt = std::move(t.startedAt);

		t.repetitions = 0ULL; // sets t as invalid
	}
	return *this;
}

double Timer::elapsed() const {
	const duration<double> elapsedS = high_resolution_clock::now() - startedAt;
	return elapsedS.count();
}
