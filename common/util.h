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

	std::chrono::duration<double> totalTime; ///< sum of all previously timed intervals

	std::chrono::time_point<std::chrono::high_resolution_clock> lastStart;	///< the start of the last timed interval

	/// how many times is the task performed before stopping the timer
	double repetitions; // kept as double, as it is used only in this form

	bool paused = true;	///< switch between pause (or not started) and resumed (or started)
	bool _done = false;	///< set by done() method, to inhibit the report from the destructor

public:
	/// Creates a timer and starts it when the parameter is true
	Timer(const std::string &taskName_, size_t repetitions_ = 1ULL, bool start = true);
	Timer(const Timer&) = delete;
	Timer(Timer &&t);
	~Timer(); ///< If not inhibited, it reports the duration of the timed process

	void operator=(const Timer&) = delete;
	Timer& operator=(Timer &&t);

	void pause();	///< Interrupts the timing until calling resume()

	void resume();	///< Resumes / starts the timing

	void done();	///< Prevents the destructor from reporting the duration of the process

	double elapsed() const;	///< @return the duration of the timed process
};

#endif // H_UTIL