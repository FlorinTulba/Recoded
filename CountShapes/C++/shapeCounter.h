/*
Part of the ShapeCounter project, which counts all possible triangles and
(convex) quadrilaterals from geometric figures traversed by a number of lines.

See the tested figures in '../TestFigures/'.

Requires Boost installation (www.boost.org).

Project compiled with g++ 5.4.0:
	g++ -I "Path_to_Boost_install_folder" -std=c++14 -fopenmp -Ofast -Wall -o countShapes
		"../../common/util.cpp" shapeCounter.cpp shapeCounter_SeqVsOpenMP.cpp main.cpp

@2017 Florin Tulba (florintulba@yahoo.com)
*/

#ifndef H_SHAPE_COUNTER
#define H_SHAPE_COUNTER

// Uncomment SHOW_CONFIG to verify the correctness of the loaded scenario
//#define SHOW_CONFIG

// Comment SHOW_SHAPES if only processing speed matters (the found shapes won't be displayed)
//#define SHOW_SHAPES

#include <vector>
#include <map>
#include <string>

#include <boost/dynamic_bitset/dynamic_bitset.hpp>

/// Counts triangles and convex quadrilaterals from a figure
class ShapeCounter {
protected:
	size_t N = 0ULL;	///< Points Count
	size_t L = 0ULL;	///< Lines Count
	size_t triangles_ = 0ULL;	///< Count of triangles
	size_t convQuadr = 0ULL;	///< Count of convex quadrilaterals

#ifdef SHOW_SHAPES
	std::vector<std::string> pointNames;	///< the names of the points
#endif // SHOW_SHAPES

	std::vector<boost::dynamic_bitset<>> lineMembers;		///< the indices of the members of each line
	std::vector<boost::dynamic_bitset<>> connections;		///< points connectivity matrix
	std::vector<boost::dynamic_bitset<>> membership;		///< membership of points to the lines
	std::vector<std::map<size_t, size_t>> membershipAsRanks;///< for each point a map between lineIdx and rankWithinLine

	typedef std::pair<size_t, size_t> Line;	///< Parameter type for the methods below providing the indices of 2 points crossed by a line

	/// @return true only if the extended lines l1 and l2 don't intersect, or intersect strictly outside the shape described by the 4 points from l1 and l2
	bool allowedIntersection(
				const Line &l1,							///< one line from a potential quadrilateral
				const Line &l2,							///< the line across from l1 in the potential quadrilateral
				const boost::dynamic_bitset<> &memL1,	///< 'and'-ed memberships (which lines include each point) of the 2 points from l1
				const boost::dynamic_bitset<> &memL2_1,	///< membership (which lines include the point) of one point from l2
				const boost::dynamic_bitset<> &memL2_2	///< membership (which lines include the point) of the other point from l2
				) const;

	/// Checks convexity of p1-p4 quadrilateral, based on the membership of each point to the available lines
	bool convex(size_t p1, const boost::dynamic_bitset<> &mem1, size_t p2, const boost::dynamic_bitset<> &mem2,
				size_t p3, const boost::dynamic_bitset<> &mem3, size_t p4, const boost::dynamic_bitset<> &mem4) const;

public:
	/**
	Configures the ShapeCounter based on the sequences of named points from the lines from the figure.
	Prepares the entire infrastructure needed while counting the shapes.
	*/
	ShapeCounter(const std::vector<std::vector<std::string>> &lines);

	/// Performs the actual shape counting in a sequential fashion
	void process();

	/// Performs the actual shape counting in parallel using OpenMP
	void process_OpenMP();

	inline size_t triangles() const { return triangles_; }
	inline size_t convexQuadrilaterals() const { return convQuadr; }
};

#endif // H_SHAPE_COUNTER