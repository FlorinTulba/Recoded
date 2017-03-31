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

/*
The implementations of the methods 'ShapeCounter::process' and 'ShapeCounter::process_OpenMP' are very similar.
Therefore, rather than duplicating their code, this file is parsed twice, first time
when COUNT_SHAPES_USING_OPEN_MP is not defined and second time after defining it.

First traversal compiles the sequential (non-OpenMP) 'ShapeCounter::process', it defines
COUNT_SHAPES_USING_OPEN_MP and it includes this file for the second parsing.
Second traversal will compile the parallel (OpenMP) 'ShapeCounter::process_OpenMP' while skipping
the code parsed during the first traversal.
*/
#ifndef COUNT_SHAPES_USING_OPEN_MP


#include "shapeCounter.h"

#include <forward_list>
#include <sstream>
#include <iostream>

#include <omp.h>

using namespace std;
using namespace boost;

/// Performs the actual shape counting in a sequential fashion
void ShapeCounter::process() {

#else // COUNT_SHAPES_USING_OPEN_MP defined

/// Performs the actual shape counting in parallel using OpenMP
void ShapeCounter::process_OpenMP() {
#ifdef SHOW_SHAPES
	forward_list<string> outputShapes; // Collecting the shapes reported by any participating thread whenever it finishes a for loop
	const auto itBeforeFirstItem = outputShapes.cbefore_begin(); // appropriate for 'outputShapes.splice_after(itLastItem)'
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
		const auto &membershipAsRanksP1 = membershipAsRanks[p1];

		// One step for ensuring the uniqueness of the solutions:
		// a mask to prevent the shapes found later from using points before P1.
		const auto maskP1 = (~dynamic_bitset<BlockType>(N)) << (p1 + 1); // Ignore connections before and including P1
		const auto connOfP1Bitset = connections[p1] & maskP1;
		const size_t countConnOfP1 = connOfP1Bitset.count();
		if(countConnOfP1 < 2ULL)
			continue; // Triangles require 2 connected points to P1. If they are not available, check next available P1

		for(size_t p2 = connOfP1Bitset.find_first(), idxP2 = 0ULL, limP2 = countConnOfP1 - 1ULL;
					idxP2 < limP2; p2 = connOfP1Bitset.find_next(p2), ++idxP2) {
			const auto &connOfP2Bitset = connections[p2];
			const auto &mem2 = membership[p2], mem1and2 = mem1 & mem2;
			const auto &membershipAsRanksP2 = membershipAsRanks[p2];
			const auto idxL12 = mem1and2.find_first();
			const auto &pointsL12 = lineMembers[idxL12];
			const size_t rankP1_L12 = membershipAsRanksP1.at(idxL12),
						rankP2_L12 = membershipAsRanksP2.at(idxL12);

#ifdef SHOW_SHAPES
			const string nameP2 = pointNames[p2];
#endif // SHOW_SHAPES

			// Points connected to P1, after P2 and not on the line P1-P2
			const auto choicesLastP =
				(connOfP1Bitset & ((~dynamic_bitset<BlockType>(N)) << (p2 + 1ULL))) - pointsL12;

			const auto choicesTriVertex = choicesLastP & connOfP2Bitset; // Points connected to P2, too
			if(choicesTriVertex.any()) {
				trCount += choicesTriVertex.count();
#ifdef SHOW_SHAPES
				for(size_t p = choicesTriVertex.find_first(); p != dynamic_bitset<>::npos; p = choicesTriVertex.find_next(p)) {
#ifdef COUNT_SHAPES_USING_OPEN_MP // Cheaper to enlist the found shape here and display the whole list at the end
					oss<<'<'<<nameP1<<nameP2<<pointNames[p]<<'>';
					itLastLocalItem = localShapes.insert_after(itLastLocalItem, oss.str());
					oss.str(""); oss.clear();
#else // COUNT_SHAPES_USING_OPEN_MP not defined - display directly the found shape
					cout<<'<'<<nameP1<<nameP2<<pointNames[p]<<"> ";
#endif // COUNT_SHAPES_USING_OPEN_MP defined or not
				}
#endif // SHOW_SHAPES
			}

			for(size_t lastP = choicesLastP.find_first(); lastP != dynamic_bitset<>::npos;
					lastP = choicesLastP.find_next(lastP)) {
#ifdef SHOW_SHAPES
				const string nameLastP = pointNames[lastP];
#endif // SHOW_SHAPES

				const auto &memLast = membership[lastP], mem1andLast = mem1 & memLast;
				const auto &membershipAsRanksLastP = membershipAsRanks[lastP];
				const auto idxL1andLast = mem1andLast.find_first();
				const size_t rankP1_L1andLast = membershipAsRanksP1.at(idxL1andLast),
						rankLastP_L1andLast = membershipAsRanksLastP.at(idxL1andLast);
				const auto &pointsL1andLast = lineMembers[idxL1andLast],
						pointsL2andLast = connOfP2Bitset[lastP] ?
							lineMembers[(mem2 & memLast).find_first()] : dynamic_bitset<BlockType>(N);

				// Points after P1, connected to P2 and LastP, but not on the line P1-P2-LastP-P1
				auto p3Choices = (connOfP2Bitset & connections[lastP] & maskP1) - 
					(pointsL12 | lineMembers[idxL1andLast] | pointsL2andLast);

				for(size_t p3 = p3Choices.find_first(); p3 != dynamic_bitset<>::npos;
						p3 = p3Choices.find_next(p3)) {
					const auto &mem3 = membership[p3];
					const auto &membershipAsRanksP3 = membershipAsRanks[p3];
					const auto idxL23 = (mem2 & mem3).find_first();
					const auto idxL3andLast = (memLast & mem3).find_first();

					// Check the intersection between L12 and L3andLast
					auto idxInters = (pointsL12 & lineMembers[idxL3andLast]).find_first();
					if(idxInters != dynamic_bitset<>::npos) {
						if( ! allowedIntersection(idxInters, idxL12, idxL3andLast,
								rankP1_L12, rankP2_L12,
								membershipAsRanksP3, membershipAsRanksLastP))
							continue; // degenerate / concave quadrilateral
					} else {
						// Check the intersection between L23 and L1andLast
						idxInters = (pointsL1andLast & lineMembers[idxL23]).find_first();
						if(idxInters != dynamic_bitset<>::npos &&
						   !allowedIntersection(idxInters, idxL1andLast, idxL23,
									rankLastP_L1andLast, rankP1_L1andLast,
									membershipAsRanksP2, membershipAsRanksP3))
							continue; // degenerate / concave quadrilateral
					}

					++quadCount;

#ifdef SHOW_SHAPES
#ifdef COUNT_SHAPES_USING_OPEN_MP // Cheaper to enlist the found shape here and display the whole list at the end
					oss<<'['<<nameP1<<nameP2<<pointNames[p3]<<nameLastP<<']';
					itLastLocalItem = localShapes.insert_after(itLastLocalItem, oss.str());
					oss.str(""); oss.clear();
#else // COUNT_SHAPES_USING_OPEN_MP not defined - display directly the found shape
					cout<<'['<<nameP1<<nameP2<<pointNames[p3]<<nameLastP<<"] ";
#endif // COUNT_SHAPES_USING_OPEN_MP defined or not
#endif // SHOW_SHAPES
				}
			}
		}

#if defined(SHOW_SHAPES) && defined(COUNT_SHAPES_USING_OPEN_MP)
		if( ! localShapes.empty()) // avoids storing below an invalid 'itLastLocalItem' and skips the critical section when there are no new shapes
		#pragma omp critical // short O(1) critical section, instead of appending outputShapes after each found shape
		{
			outputShapes.splice_after(itBeforeFirstItem, std::move(localShapes));
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
// Do not include 'shapeCounter_SeqVsOpenMP.cpp' in a different 'cpp' file! Otherwise, __FILE__ points to the enclosing 'cpp' file.

// The first traversal resumes from here

#endif // COUNT_SHAPES_USING_OPEN_MP
