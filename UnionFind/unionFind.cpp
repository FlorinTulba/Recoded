/*
    Implementation of the UnionFind data structure described here:
        https://en.wikipedia.org/wiki/Disjoint-set_data_structure

    @2017 Florin Tulba (florintulba@yahoo.com)
*/


#include <iostream>
#include <iterator>
#include <vector>
#include <map>

using namespace std;

/// Data specific for each item of the Union Find
struct UfItem {
	int ancestor; ///< the most distant known ancestor
	int rank = 0; ///< initial depth of a subtree rooted on this item (some children might migrate closer to the root of this item)

	UfItem(int id) : ancestor(id) {}
};

/// The UnionFind class
class UF {
protected:
	vector<UfItem> items;	///< the managed items
	int groups;				///< current count of the groups formed from the items

public:
	/// Create itemCount items that are initially separated
	UF(int itemCount) : groups(itemCount) {
		items.reserve(itemCount);
		for(int i=0; i<itemCount; ++i)
			items.emplace_back(i);
	}

	/// Find parent operation
	int parentOf(int id) {
		int *parentId = nullptr;
		while(*(parentId = &items[id].ancestor) != id)
			id = *parentId = items[*parentId].ancestor;

		return id;
	}

	/// Are items id1 and id2 connected ?
	bool connected(int id1, int id2) {
		return parentOf(id1) == parentOf(id2);
	}

	/// Connect id1 & id2
	void join(int id1, int id2) {
		id1 = parentOf(id1); id2 = parentOf(id2);
		if(id1 == id2)
			return;

		const int rank1 = items[id1].rank, rank2 = items[id2].rank;
		if(rank1 < rank2)
			items[id1].ancestor = id2;
		else
			items[id2].ancestor = id1;
		
		if(rank1 == rank2)
			++items[id1].rank;
		
		--groups;
	}

	friend ostream& operator<<(ostream &os, UF &uf) {
		os<<uf.groups<<" groups: ";
		map<int, vector<int>> mapping;
		for(int i=0, lim = uf.items.size(); i<lim; ++i)
			mapping[uf.parentOf(i)].push_back(i);
		for(const auto &group: mapping) {
			os<<group.first<<'{';
			copy(begin(group.second),end(group.second), ostream_iterator<int>(os, " "));
			os<<"}  ";
		}
		return os;
	}
};

int main() {
	UF uf(10);
	cout<<"Initial uf is:"<<endl<<uf<<endl;
	// uf.join(2, 1);
	// uf.join(0, 4);
	// uf.join(0, 5);
	// uf.join(6, 3);
	// uf.join(7, 4);
	// uf.join(9, 4);
	// uf.join(2, 6);
	// uf.join(2, 5);
	// uf.join(1, 8);
	// cout<<uf<<endl;

	uf.join(0, 3);
	cout<<uf<<endl;
	uf.join(4, 5);
	cout<<uf<<endl;
	uf.join(1, 9);
	cout<<uf<<endl;
	uf.join(2, 8);
	cout<<uf<<endl;
	uf.join(7, 4);
	cout<<uf<<endl;
	uf.join(9, 0);
	cout<<uf<<endl;
	uf.join(7, 8);
	cout<<uf<<endl;
	uf.join(1, 6);
	cout<<uf<<endl;
	uf.join(0, 5);
	cout<<uf<<endl;

	return 0;
}