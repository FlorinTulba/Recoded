/*
Implementation of the UnionFind data structure described here:
https://en.wikipedia.org/wiki/Disjounsigned-set_data_structure

Compiled with g++ 5.4.0:
	g++ -std=c++11 -Ofast -Wall -o unionFind unionFind.cpp

@2017 Florin Tulba (florintulba@yahoo.com)
*/

#include <string>
#include <fstream>
#include <sstream>
#include <iostream>
#include <iomanip>
#include <iterator>
#include <vector>
#include <map>
#include <memory>
#include <cassert>

using namespace std;

/// Data specific for each item of the Union Find
struct UfItem {
	unsigned ancestor;	///< the most distant known ancestor
	unsigned rank = 0U;	///< initial depth of a subtree rooted on this item (some children might migrate closer to the root of this item)

	UfItem(unsigned id) : ancestor(id) {}
};

/// The UnionFind class. It's the one in charge of validating the data input
class UF {
protected:
	vector<UfItem> items;	///< the managed items
	unsigned itemCount;		///< size of items
	unsigned groups;		///< current count of the groups formed from the items

	/// Checks if the provided index might be a valid element index
	bool validateIndex(unsigned idx) const {
		if(idx < itemCount)
			return true;

		cerr<<"Invalid element index: "<<idx<<endl;
		return false;
	}

	/// Find parent operation
	unsigned parentOf(unsigned id) {
		assert(validateIndex(id));

		unsigned *parentId = nullptr;
		while(*(parentId = &items[id].ancestor) != id)
			id = *parentId = items[*parentId].ancestor;

		return id;
	}

public:
	/// Create itemCount items that are initially separated
	UF(unsigned itemCount_) : itemCount(itemCount_), groups(itemCount_) {
		items.reserve(itemCount);
		for(unsigned i = 0U; i<itemCount; ++i)
			items.emplace_back(i);

		cout<<" Initially: "<<*this<<endl;
		if(itemCount < 2U)
			cout<<"Note that this problem makes sense only for at least 2 elements!"<<endl;
	}

	/// Are items id1 and id2 connected ?
	bool connected(unsigned id1, unsigned id2) {
		if( ! validateIndex(id1) || ! validateIndex(id2))
			return false;

		return parentOf(id1) == parentOf(id2);
	}

	/// Connect id1 & id2
	void join(unsigned id1, unsigned id2) {
		cout<<setw(3)<<id1<<" - "<<setw(3)<<id2<<" : ";
		if( ! validateIndex(id1) || ! validateIndex(id2))
			return;

		id1 = parentOf(id1); id2 = parentOf(id2);
		if(id1 == id2) {
			cout<<*this<<endl;
			return;
		}

		const unsigned rank1 = items[id1].rank, rank2 = items[id2].rank;
		if(rank1 < rank2)
			items[id1].ancestor = id2;
		else
			items[id2].ancestor = id1;

		if(rank1 == rank2)
			++items[id1].rank;

		--groups;

		cout<<*this<<endl;
		if(groups == 1U)
			cout<<"All elements are now connected!"<<endl;
	}

	friend ostream& operator<<(ostream &os, UF &uf) {
		os<<setw(3)<<uf.groups<<" groups: ";
		map<unsigned, vector<unsigned>> mapping;
		for(unsigned i = 0U, lim = uf.itemCount; i<lim; ++i)
			mapping[uf.parentOf(i)].push_back(i);
		for(const auto &group: mapping) {
			os<<group.first<<'{';
			copy(begin(group.second), end(group.second), ostream_iterator<unsigned>(os, " "));
			os<<"\b}  ";
		}
		return os;
	}
};

/// Provides the problem input data
class ScenarioProvider {
protected:
	ifstream ifs;			///< stream providing the test scenario
	unsigned items = 0U;	///< number of elements from the described scenario
	bool valid = true;		///< becomes false as soon as IO / parsing errors are found in the scenario file

public:
	typedef pair<unsigned, unsigned> ElemPair;	///< the indices from a pair of elements

	/// Reads itemsCount from 'testScenario.txt'
	ScenarioProvider() {
		static const string scenarioFile("testScenario.txt");
		ifs.open(scenarioFile);
		if( ! ifs) {
			cerr<<"Couldn't open the scenario file: "<<scenarioFile<<endl;
			valid = false;
			return;
		}
		string line;
		while(getline(ifs, line)) {
			if((line.empty() || (line.size() == 1ULL && line[0] == '\r')) || line[0] == '#')
				continue;	// Ignore empty lines or lines containing comments (these start with '#')

			istringstream iss(line);
			if( ! (iss>>items)) {
				cerr<<"Couldn't read the items count from the scenario file! Please correct this error and then try again!"<<endl;
				valid = false;
			}

			return;
		}
	}

	inline unsigned itemsCount() const { return items; }

	/// Reads next pair of element indices
	unique_ptr<ElemPair> nextElemPair() {
		if( ! valid) {
			cout<<"Please correct previous errors before calling ScenarioProvider::nextElemPair()!"<<endl;
			return nullptr;
		}

		if(ifs.eof()) {
			cout<<"No other pairs to connect in this scenario!"<<endl;
			return nullptr;
		}

		string line;
		while(getline(ifs, line)) {
			if((line.empty() || (line.size() == 1ULL && line[0] == '\r')) || line[0] == '#')
				continue;	// Ignore empty lines or lines containing comments (these start with '#')

			istringstream iss(line);
			unsigned idx1, idx2;
			if( ! (iss>>idx1>>idx2)) {
				cerr<<"Couldn't read the next pair of element indices from the scenario file! Please correct this error and then try again!"<<endl;
				valid = false;
				break;
			}

			//return make_unique<ElemPair>(idx1, idx2); // C++14
			return unique_ptr<ElemPair>(new ElemPair(idx1, idx2));
		}

		return nullptr;
	}
};

int main() {
	ScenarioProvider sp;
	UF uf(sp.itemsCount());
	unique_ptr<ScenarioProvider::ElemPair> elemPair;
	while(nullptr != (elemPair = sp.nextElemPair()))
		uf.join(elemPair->first, elemPair->second);

	return 0;
}