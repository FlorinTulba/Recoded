/*
Counting all possible triangles and (convex) quadrilaterals from geometric figures traversed by a number of lines.
See the tested figures in '../TestFigures/'.

Requires Boost installation (www.boost.org).

Compiled with g++ 5.4.0:
	g++ -I "Path_to_Boost_install_folder" -std=c++14 -fopenmp -Ofast -Wall "../../common/util.cpp" countShapes.cpp -o countShapes

@2017 Florin Tulba (florintulba@yahoo.com)
*/

/*
Counting the shapes is performed in the methods 'ShapeCounter::process' and 'ShapeCounter::process_OpenMP',
whose implementations look very similar. Therefore, rather than duplicating their code, this file is parsed
twice, first time when COUNT_SHAPES_USING_OPEN_MP is not defined and second time after defining it.

First traversal compiles the sequential (non-OpenMP) 'ShapeCounter::process', it defines
COUNT_SHAPES_USING_OPEN_MP and it includes this file for the second parsing.
Second traversal will compile the parallel (OpenMP) 'ShapeCounter::process_OpenMP' while skipping
the rest of the code parsed during the first traversal.
*/
#ifndef COUNT_SHAPES_USING_OPEN_MP

//#define SHOW_CONFIG			// Uncomment to verify the correctness of the loaded scenario
#define SHOW_SHAPES				// Comment if only processing speed matters (the found shapes won't be displayed)

#include "../../common/util.h"

#include <forward_list>
#include <vector>
#include <set>
#include <map>
#include <string>
#include <iostream>
#include <sstream>
#include <fstream>
#include <algorithm>
#include <iterator>

#include <omp.h>

#include <boost/dynamic_bitset/dynamic_bitset.hpp>

using namespace std;
using namespace boost;

/// Counts triangles and convex quadrilaterals from a figure
class ShapeCounter {
protected:
	size_t N = 0ULL;	///< Points Count
	size_t L = 0ULL;	///< Lines Count
	size_t triangles_ = 0ULL;	///< Count of triangles
	size_t convQuadr = 0ULL;	///< Count of convex quadrilaterals

#ifdef SHOW_SHAPES
	vector<string> pointNames;			///< the names of the points
#endif // SHOW_SHAPES

	vector<dynamic_bitset<>> lineMembers;		///< the indices of the members of each line
	vector<dynamic_bitset<>> connections;		///< points connectivity matrix
	vector<dynamic_bitset<>> membership;		///< membership of points to the lines
	vector<map<size_t, size_t>> membershipAsRanks; ///< for each point a map between lineIdx and rankWithinLine

	typedef pair<size_t, size_t> Line;	///< Parameter type for the methods below providing the indices of 2 points crossed by a line

	/// @return true only if the extended lines l1 and l2 don't intersect, or intersect strictly outside the shape described by the 4 points from l1 and l2
	bool allowedIntersection(const Line &l1,					///< one line from a potential quadrilateral
							const Line &l2,						///< the line across from l1 in the potential quadrilateral
							const dynamic_bitset<> &memL1,		///< 'and'-ed memberships (which lines include each point) of the 2 points from l1
							const dynamic_bitset<> &memL2_1,	///< membership (which lines include the point) of one point from l2
							const dynamic_bitset<> &memL2_2		///< membership (which lines include the point) of the other point from l2
							) const {
		const size_t lineIdxPair1 = memL1.find_first();
		if(memL2_1[lineIdxPair1] || memL2_2[lineIdxPair1])
			return false; // one of the provided points from L2 are inside L1

		const size_t lineIdxPair2 = (memL2_1 & memL2_2).find_first();
		const auto intersectionPoint = (lineMembers[lineIdxPair1] & lineMembers[lineIdxPair2]).find_first();
		if(intersectionPoint != dynamic_bitset<>::npos) {
			// The found intersection point should fall outside the segment l1
			// The check relies on the fact that lines specify the contained points in order
			size_t rank1 = membershipAsRanks[l1.first].find(lineIdxPair1)->second,
					rank2 = membershipAsRanks[l1.second].find(lineIdxPair1)->second;
			if(rank1 > rank2)
				swap(rank1, rank2);
			const auto &intersectionPointMembership = membershipAsRanks[intersectionPoint];
			auto rank = intersectionPointMembership.find(lineIdxPair1)->second;
			if(rank1 <= rank && rank <= rank2)
				return false;

			// The found intersection point should fall outside the segment l2
			rank1 = membershipAsRanks[l2.first].find(lineIdxPair2)->second;
			rank2 = membershipAsRanks[l2.second].find(lineIdxPair2)->second;
			if(rank1 > rank2)
				swap(rank1, rank2);
			rank = intersectionPointMembership.find(lineIdxPair2)->second;
			if(rank1 <= rank && rank <= rank2)
				return false;
		}

		return true; // no intersection or the intersection is 'external'
	}

	/// Checks convexity of p1-p4 quadrilateral, based on the membership of each point to the available lines
	bool convex(size_t p1, const dynamic_bitset<> &mem1, size_t p2, const dynamic_bitset<> &mem2,
				size_t p3, const dynamic_bitset<> &mem3, size_t p4, const dynamic_bitset<> &mem4) const {
		assert(max(p1, p2) < N && max(p3, p4) < N);
		assert(p1!=p2 && p1!=p3 && p1!=p4 && p2!=p3 && p2!=p4 && p3!=p4);

		// Extended p1-p2 and p3-p4 shouldn't touch
		if(!allowedIntersection(Line(p1, p2), Line(p3, p4), (mem1 & mem2), mem3, mem4))
		   return false;

		// Extended p2-p3 and p4-p1 shouldn't touch
		if(!allowedIntersection(Line(p2, p3), Line(p4, p1), (mem2 & mem3), mem4, mem1))
			return false;

		return true;
	}

public:
	/**
	Configures the ShapeCounter based on the sequences of named points from the lines from the figure.
	Prepares the entire infrastructure needed while counting the shapes.
	*/
	ShapeCounter(const vector<vector<string>> &lines) {
		lineMembers.reserve(L = lines.size());
		map<string, size_t> pointsIndices;
		for(size_t lineIdx = 0ULL; lineIdx < L; ++lineIdx) {
			const auto &line = lines[lineIdx];
			const auto pointsOnLine = line.size();
			vector<size_t> memberIndices; memberIndices.reserve(pointsOnLine);
			dynamic_bitset<> memberIndicesBitset(N);
			for(size_t pointRank = 0ULL; pointRank < pointsOnLine; ++pointRank) {
				const auto &pointName = line[pointRank];
				size_t pointIdx;
				const auto it = pointsIndices.find(pointName);
				if(it == pointsIndices.cend()) {

#ifdef SHOW_SHAPES
					pointNames.push_back(pointName);
#endif // SHOW_SHAPES

					pointIdx = pointsIndices[pointName] = N++;
					for(auto &prevMembers : lineMembers)
						prevMembers.push_back(false);
					for(auto &conns : connections)
						conns.push_back(false);
					connections.emplace_back(N);
					memberIndicesBitset.push_back(true);
					membership.emplace_back(L);
					membershipAsRanks.emplace_back();
				} else {
					pointIdx = it->second;
					memberIndicesBitset[pointIdx] = true;
				}
				auto &lastPointConns = connections[pointIdx];
				for(auto prevIdx : memberIndices) {
					lastPointConns[prevIdx] = connections[prevIdx][pointIdx] = true;
				}
				memberIndices.push_back(pointIdx);
				membership[pointIdx][lineIdx] = true;
				membershipAsRanks[pointIdx][lineIdx] = pointRank;
			}
			lineMembers.emplace_back(memberIndicesBitset);
		}

#ifdef SHOW_CONFIG
		for(size_t i = 0ULL; i < N; ++i) {
			cout<<
#ifdef SHOW_SHAPES
				pointNames[i]
#else // SHOW_SHAPES
				"Point "<<i
#endif // SHOW_SHAPES
				<<": connections {"<<connections[i]<<"} ; member of lines {"<<membership[i]<<"} ; pos in lines {";
			for(const auto &lineIdxRankPair : membershipAsRanks[i])
				cout<<lineIdxRankPair.second<<"(l"<<lineIdxRankPair.first<<") ";

			cout<<"\b}"<<endl;
		}

		for(size_t i = 0ULL; i < L; ++i)
			cout<<'L'<<i<<": members {"<<lineMembers[i]<<'}'<<endl;

		cout<<endl;
#endif // SHOW_CONFIG
	}

	/// Performs the actual shape counting in a sequential fashion
	void process() {
#else // COUNT_SHAPES_USING_OPEN_MP defined
	/// Performs the actual shape counting in parallel using OpenMP
	void process_OpenMP() {
#ifdef SHOW_SHAPES
		forward_list<string> outputShapes; // Collecting the shapes reported by any participating thread whenever it finishes a for loop
		auto itLastItem = outputShapes.cbefore_begin(); // appropriate for 'outputShapes.splice_after(itLastItem)'
#endif // SHOW_SHAPES
#endif // COUNT_SHAPES_USING_OPEN_MP defined or not

		// Count of each type of shape to be used within OpenMP reduction and
		// to be copied over the class fields at the end of the for loop
		size_t trCount = 0ULL, quadCount = 0ULL;
		
		// Total for loops to be dynamically distributed among the participating threads
		const int limP1 = int(N) - 2;
#ifdef COUNT_SHAPES_USING_OPEN_MP // more efficient than '#pragma omp parallel if(false)'
		#pragma omp parallel
		#pragma omp for schedule(dynamic) nowait reduction(+ : trCount, quadCount)
#endif // COUNT_SHAPES_USING_OPEN_MP
		for(int p1 = 0; p1 < limP1; ++p1) {

#ifdef SHOW_SHAPES
#ifdef COUNT_SHAPES_USING_OPEN_MP
			forward_list<string> localShapes; // Collects all the shapes generated starting from point p1
			auto itLastLocalItem = localShapes.cbefore_begin(); // appropriate for 'localShapes.insert_after(itLastLocalItem)'
			ostringstream oss; // shape names build helper
#endif // COUNT_SHAPES_USING_OPEN_MP
			const string nameP1 = pointNames[p1];
#endif // SHOW_SHAPES

			const auto &mem1 = membership[p1];

			// One step for ensuring the uniqueness of the solutions:
			// a mask to prevent the shapes found later from using points before P1.
			const auto maskP1 = (~dynamic_bitset<>(N)) << (p1+1); // Ignore connections before and including P1
			const auto connOfP1Bitset = connections[p1] & maskP1;
			const size_t countConnOfP1 = connOfP1Bitset.count();
			if(countConnOfP1 < 2ULL)
				continue; // Triangles require 2 connected points to P1. If they are not available, check next available P1

			vector<size_t> connOfP1; connOfP1.reserve(countConnOfP1);
			for(size_t p = connOfP1Bitset.find_first(), idx = 0ULL; idx < countConnOfP1; p = connOfP1Bitset.find_next(p), ++idx)
				connOfP1.push_back(p);
			
			for(size_t idxP2 = 0ULL, p2 = connOfP1.front(), limP2 = countConnOfP1 - 1ULL; idxP2 < limP2; p2 = connOfP1[++idxP2]) {
				const auto &mem2 = membership[p2], mem1and2 = mem1 & mem2;

#ifdef SHOW_SHAPES
				const string nameP2 = pointNames[p2];
#endif // SHOW_SHAPES

				for(size_t idxLastP = idxP2 + 1ULL; idxLastP < countConnOfP1; ++idxLastP) {
					const size_t lastP = connOfP1[idxLastP];
					const auto &memLast = membership[lastP];
					if((mem1and2 & memLast).any()) // coll(p1, p2, lastP)
						continue;	// Ignore collinear points

#ifdef SHOW_SHAPES
					const string nameLastP = pointNames[lastP];
#endif // SHOW_SHAPES

					if(connections[p2][lastP]) {
						++trCount;
#ifdef SHOW_SHAPES
#ifdef COUNT_SHAPES_USING_OPEN_MP // Cheaper to enlist the found shape here and display the whole list at the end
						oss<<'<'<<nameP1<<nameP2<<nameLastP<<'>';
						itLastLocalItem = localShapes.insert_after(itLastLocalItem, oss.str());
						oss.str(""); oss.clear();
#else // COUNT_SHAPES_USING_OPEN_MP not defined - display directly the found shape
						cout<<'<'<<nameP1<<nameP2<<nameLastP<<"> ";
#endif // COUNT_SHAPES_USING_OPEN_MP defined or not
#endif // SHOW_SHAPES
					}

					const auto connOfP2_LastP_Bitset = connections[p2] & connections[lastP] & maskP1;
					const auto mem1and2or2andLast = mem1and2 | (mem2 & memLast);
					for(size_t p3 = connOfP2_LastP_Bitset.find_first(); p3 != dynamic_bitset<>::npos; p3 = connOfP2_LastP_Bitset.find_next(p3)) {
						const auto &mem3 = membership[p3];
						if((mem1and2or2andLast & mem3).any()) // coll(p1, p2, p3) || coll(lastP, p2, p3)
							continue;	// Ignore collinear points

						if(convex(p1, mem1, p2, mem2, p3, mem3, lastP, memLast)) {
							++quadCount;
#ifdef SHOW_SHAPES
#ifdef COUNT_SHAPES_USING_OPEN_MP // Cheaper to enlist the found shape here and display the whole list at the end
							oss<<'['<<nameP1<<nameP2<<pointNames[p3]<<nameLastP<<']';
							itLastLocalItem = localShapes.insert_after(itLastLocalItem, oss.str());
							oss.str(""); oss.clear();
#else // COUNT_SHAPES_USING_OPEN_MP not defined - display directly the found shape
							cout<<'['<<nameP1<<nameP2<<pointNames[p3]<<nameLastP<<"> ";
#endif // COUNT_SHAPES_USING_OPEN_MP defined or not
#endif // SHOW_SHAPES
						}
					}
				}
			}

#if defined(SHOW_SHAPES) && defined(COUNT_SHAPES_USING_OPEN_MP)
			if( ! localShapes.empty()) // avoids storing below an invalid 'itLastLocalItem' and skips the critical section when there are no new shapes
			#pragma omp critical // short O(1) critical section, instead of appending outputShapes after each found shape
			{
				outputShapes.splice_after(itLastItem, std::move(localShapes));
				itLastItem = itLastLocalItem;
			}
#endif // SHOW_SHAPES && COUNT_SHAPES_USING_OPEN_MP
		} // #pragma omp for    ends here

		triangles_ = trCount;
		convQuadr = quadCount;

#ifdef SHOW_SHAPES
#ifdef COUNT_SHAPES_USING_OPEN_MP
		copy(cbegin(outputShapes), cend(outputShapes), ostream_iterator<string>(cout, " "));
#endif // COUNT_SHAPES_USING_OPEN_MP
		cout<<endl;
#endif // SHOW_SHAPES
	}

#ifndef COUNT_SHAPES_USING_OPEN_MP
#	define COUNT_SHAPES_USING_OPEN_MP
#	include __FILE__	// Produces the second traversal of the file, with COUNT_SHAPES_USING_OPEN_MP defined this time
	// Do not include 'countShapes.cpp' in a different 'cpp' file! Otherwise, __FILE__ points to the enclosing 'cpp' file.

	// The first traversal resumes from here and ends at the end of the file

	size_t triangles() const { return triangles_; }
	size_t convexQuadrilaterals() const { return convQuadr; }
};

int main() {
	// Select one of the following scenario files (uncommenting it and commenting the others)
	// or create a new scenario and open your own file:

	// Scenario from figure 'count6Shapes.png'
	const string testFileName = "../TestFigures/TextVersions/count6shapes.txt";		// Manually generated (same labels as in the corresponding figure)
	//const string testFileName = "../TestFigures/TextVersions/count6shapes.png.txt";	// Generated by the figure interpreter written in Matlab/Octave

	// Scenario from figure 'count9Shapes.png'
	//const string testFileName = "../TestFigures/TextVersions/count9shapes.txt";		// Manually generated (same labels as in the corresponding figure)
	//const string testFileName = "../TestFigures/TextVersions/count9shapes.png.txt";	// Generated by the figure interpreter written in Matlab/Octave

	// Scenario from figure 'count100Shapes.png'
	//const string testFileName = "../TestFigures/TextVersions/count100shapes.txt";	// Manually generated (same labels as in the corresponding figure)
	//const string testFileName = "../TestFigures/TextVersions/count100shapes.png.txt";// Generated by the figure interpreter written in Matlab/Octave

	// Scenario from figure 'count673Shapes.png'
	//const string testFileName = "../TestFigures/TextVersions/count673shapes.txt";	// Manually generated (same labels as in the corresponding figure)
	//const string testFileName = "../TestFigures/TextVersions/count673shapes.png.txt";// Generated by the figure interpreter written in Matlab/Octave

	ifstream ifs(testFileName);
	if( ! ifs) {
		cerr<<"Couldn't open file '"<<testFileName<<'\''<<endl;
		return -1;
	}

	vector<vector<string>> lines;
	string line;
	while(nextRelevantLine(ifs, line)) {
		istringstream iss(line);
		lines.emplace_back(istream_iterator<string>(iss), istream_iterator<string>());
	}

	ifs.close();

	ShapeCounter sc(lines);

	sc.process();
	size_t totalShapes = sc.triangles() + sc.convexQuadrilaterals();
	cout<<"There are "<<sc.triangles()<<" triangles and "<<sc.convexQuadrilaterals()
		<<" convex quadrilaterals, which means "<<totalShapes<<" convex shapes in total."<<endl<<endl;

	sc.process_OpenMP();
	totalShapes = sc.triangles() + sc.convexQuadrilaterals();
	cout<<"There are "<<sc.triangles()<<" triangles and "<<sc.convexQuadrilaterals()
		<<" convex quadrilaterals, which means "<<totalShapes<<" convex shapes in total."<<endl<<endl;

	return 0;
}
#endif // COUNT_SHAPES_USING_OPEN_MP
