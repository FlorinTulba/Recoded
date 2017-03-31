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
	lineMembersByRank.reserve(L);
	map<string, size_t> pointsIndices;
	for(size_t lineIdx = 0ULL; lineIdx < L; ++lineIdx) {
		const auto &line = lines[lineIdx];
		const auto pointsOnLine = line.size();
		vector<size_t> memberIndices; memberIndices.reserve(pointsOnLine);
		dynamic_bitset<BlockType> memberIndicesBitset(N);
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
		lineMembersByRank.emplace_back(memberIndices);
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
			cout<<lineIdxRankPair.second<<"(L"<<lineIdxRankPair.first<<") ";

		cout<<"\b}"<<endl;
	}

	for(size_t i = 0ULL; i < L; ++i) {
		cout<<'L'<<i<<": members {"<<lineMembers[i]<<" that is points ";
		for(const auto pIdx : lineMembersByRank[i]) {
			cout<<
#ifdef SHOW_SHAPES
				pointNames[pIdx]
#else // SHOW_SHAPES
				pIdx
#endif // SHOW_SHAPES
				<<' ';
		}
		cout<<"\b}"<<endl;
	}

	cout<<endl;
#endif // SHOW_CONFIG
}

bool ShapeCounter::allowedIntersection(size_t intersectionPoint,
									   size_t idxL12, size_t idxL34,
									   size_t rankP1_L12, size_t rankP2_L12,
									   const LineIdx_Rank &mp3, const LineIdx_Rank &mp4) const {
	// The found intersection point should fall outside the segment L12
	// The check relies on the fact that lines specify the contained points in order

	const auto &intersectionPointMembership = membershipAsRanks[intersectionPoint];
	auto rank = intersectionPointMembership.at(idxL12);
	bool r1LessR2 = rankP1_L12 < rankP2_L12,
		rLessR1 = rank < rankP1_L12,
		rLessR2 = rank < rankP2_L12;

	// When caseR12or21R is true, L12 contains either the sequence X-P1-P2 or P2-P1-X,
	// where X is the intersection point
	bool caseR12or21R = false;
	if(r1LessR2) {
		if(rLessR1) caseR12or21R = true;
		else if(rLessR2) return false;
	} else {
		if( ! rLessR1) caseR12or21R = true;
		else if( ! rLessR2) return false;
	}

	// The found intersection point should fall outside the segment L34
	// and on the side imposed by the boolean caseR12or21R
	size_t rank1 = mp3.at(idxL34), // P3 as if P1 and P4 as if P2, to use same notations as above
		rank2 = mp4.at(idxL34);
	rank = intersectionPointMembership.at(idxL34);
	r1LessR2 = rank1 < rank2;
	rLessR1 = rank < rank1;
	rLessR2 = rank < rank2;
	if(r1LessR2) {
		if(caseR12or21R) return ( ! rLessR2);
		return rLessR1;
	}

	if(caseR12or21R) return rLessR2;
	return ( ! rLessR1);
}
