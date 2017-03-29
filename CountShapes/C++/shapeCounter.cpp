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

#include "shapeCounter.h"

#include <iostream>

using namespace std;
using namespace boost;

ShapeCounter::ShapeCounter(const vector<vector<string>> &lines) {
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


bool ShapeCounter::allowedIntersection(
					const Line &l1, const Line &l2,
					const dynamic_bitset<> &memL1,
					const dynamic_bitset<> &memL2_1,
					const dynamic_bitset<> &memL2_2) const {
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


bool ShapeCounter::convex(
					size_t p1, const dynamic_bitset<> &mem1, size_t p2, const dynamic_bitset<> &mem2,
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
