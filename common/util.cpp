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

Timer::Timer(const std::string &taskName_, size_t repetitions_/* = 1ULL*/, bool start/* = true*/) :
		taskName(taskName_),
		repetitions((double)repetitions_) {
	if(0ULL == repetitions_) {
		static const string err("Don't construct a Timer with repetitions_ = 0!");
		cerr<<err<<endl;
		throw invalid_argument(err);
	}
	if(start)
		resume();
}

Timer::Timer(Timer &&t) :
		taskName(std::move(t.taskName)),
		totalTime(std::move(t.totalTime)),
		lastStart(std::move(t.lastStart)),
		repetitions(t.repetitions),
		paused(t.paused),
		_done(t._done) {
	t._done = true; // prevents a second time report issued when t is destructed
}

Timer::~Timer() {
	if(!_done) {
		const double elapsedS = elapsed();
		cout<<"Task '"<<taskName<<"' required: "<< elapsedS / repetitions <<"s!"<<endl;
	}
}

Timer& Timer::operator=(Timer &&t) {
	if(this != &t) {
		taskName = std::move(t.taskName);
		totalTime = std::move(t.totalTime);
		lastStart = std::move(t.lastStart);
		repetitions = t.repetitions;
		paused = t.paused;
		_done = t._done;

		t._done = true; // prevents a second time report issued when t is destructed
	}
	return *this;
}

double Timer::elapsed() const {
	if(paused)
		return totalTime.count();

	return (totalTime + high_resolution_clock::now() - lastStart).count();
}

void Timer::pause() {
	if(!paused) {
		totalTime += high_resolution_clock::now() - lastStart;
		paused = true;
	}
}

void Timer::resume() {
	if(paused) {
		paused = false;
		lastStart = high_resolution_clock::now();
	}
}

void Timer::done() {
	_done = true;
}
