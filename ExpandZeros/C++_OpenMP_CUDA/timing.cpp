/*
Given a matrix, for any element with value 0,
set all elements from the corresponding row and column on 0, as well.

Implementations using OpenMP and CUDA for NVIDIA GPUs.

@2017 Florin Tulba (florintulba@yahoo.com)
*/

#include "timing.h"

#include <iostream>

using namespace std;
using namespace std::chrono;

Timer::Timer(bool start/* = true*/) {
	if(start)
		resume();
}

Timer::~Timer() {
	if(!done_) {
		const double report = elapsed();
		cout<<"The process took"<<report<<"s!"<<endl;
	}
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
	done_ = true;
}

double Timer::elapsed() const {
	if(paused)
		return totalTime.count();

	return (totalTime + high_resolution_clock::now() - lastStart).count();
}
