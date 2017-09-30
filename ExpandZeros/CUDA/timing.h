/*
Given a matrix, for any element with value 0,
set all elements from the corresponding row and column on 0, as well.

Implementation using CUDA for NVIDIA GPUs.

@2017 Florin Tulba (florintulba@yahoo.com)
*/

#ifndef H_TIMING
#define H_TIMING

#include <chrono>

class Timer {
protected:
	std::chrono::duration<double> totalTime; ///< sum of all previously timed intervals

	/// the start of the last timed interval
	std::chrono::time_point<std::chrono::high_resolution_clock> lastStart;

	bool paused = true;	///< switch between pause (or not started) and resumed (or started)
	bool done_ = false;	///< set by done() method, to inhibit the report from the destructor

public:
	/// Creates a timer and starts it when the parameter is true
	Timer(bool start = true);

	/// If not inhibited, it reports the duration of the timed process
	~Timer();

	void pause();	///< Interrupts the timing until calling resume()

	void resume();	///< Resumes / starts the timing

	void done();	///< Prevents the destructor from reporting the duration of the process

	double elapsed() const;	///< @return the duration of the timed process
};

#endif // H_TIMING