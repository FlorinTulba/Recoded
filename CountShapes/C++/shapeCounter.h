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

	// When BlockType = size_t, OpenMP version reaches top performance, while the sequential version is worse.
	// When BlockType = unsigned long, both versions perform quite well (optimal for the sequential version)
	// Both versions' performance degrades for unsigned short or unsigned char
	typedef unsigned long BlockType;

	std::vector<boost::dynamic_bitset<BlockType>> lineMembers;	///< the indices of the members of each line
	std::vector<boost::dynamic_bitset<BlockType>> connections;	///< points connectivity matrix
	std::vector<boost::dynamic_bitset<BlockType>> membership;	///< membership of points to the lines

	typedef std::map<size_t, size_t> LineIdx_Rank;	///< map between lineIdx and rankWithinLine
	std::vector<LineIdx_Rank> membershipAsRanks;	///< for each point a map between lineIdx and rankWithinLine

	/// @return true only if the provided intersection point denotes a non-degenerate and non-concave quadrilateral
	bool allowedIntersection(size_t intersectionPoint,	///< index of the intersection point
							 size_t idxL12,				///< index of the line passing through the first 2 considered points
							 size_t idxL34,				///< index of the line passing through the last 2 considered points
							 size_t rankP1_L12,			///< rank of P1 within line L12
							 size_t rankP2_L12,			///< rank of P2 within line L12
							 const LineIdx_Rank &mp3,	///< a map for the third point with the lines it belongs to and its rank on each such line
							 const LineIdx_Rank &mp4	///< a map for the forth point with the lines it belongs to and its rank on each such line
							 ) const;

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