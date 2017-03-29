/*
	Various utilities shared by several 'Recoded' projects in C++

	@2017 Florin Tulba (florintulba@yahoo.com)
*/

#ifndef H_UTIL
#define H_UTIL

#include <string>
#include <istream>
#include <chrono>

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


/// Simple timer class to measure the performance of some task
class Timer {
protected:
	std::string taskName;			///< name of the monitored task

	/// how many times is the task performed before stopping the timer. 0 if *this is moved to a different Timer
	size_t repetitions;

	std::chrono::time_point<std::chrono::high_resolution_clock> startedAt;	///< starting moment

public:
	Timer(const std::string &taskName_, size_t repetitions_ = 1ULL); // Starts the timer
	Timer(const Timer&) = delete;
	Timer(Timer &&t);
	~Timer(); ///< Reports (average) elapsed time for a valid Timer (repetitions != 0ULL)

	void operator=(const Timer&) = delete;
	Timer& operator=(Timer &&t);

	/// Returns the time elapse since the Timer was started (created)
	double elapsed() const;
};

#endif // H_UTIL