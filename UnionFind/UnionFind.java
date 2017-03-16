/*
    Implementation of the UnionFind data structure described here:
        https://en.wikipedia.org/wiki/Disjoint-set_data_structure

    @2017 Florin Tulba (florintulba@yahoo.com)
*/


import java.util.*;

public class UnionFind {
	// Data specific for each item of the Union Find
	protected class UfItem {
		public int ancestor;	// the most distant known ancestor
		public int rank;		// initial depth of a subtree rooted on this item (some children might migrate closer to the root of this item)
		public UfItem(int id) {
			ancestor = id;
			rank = 0;
		}
	}

	protected UfItem[] items;	// the managed items
	public int groups;			// current count of the groups formed from the items

	public UnionFind(int itemsCount) {
		groups = itemsCount;
		items = new UfItem[itemsCount];
		for(int i=0; i<itemsCount; ++i)
			items[i] = new UfItem(i);
	}

	public int parentOf(int id) {
		int parentId;
		while(id != (parentId = items[id].ancestor))
			id = items[id].ancestor = items[parentId].ancestor;
		
		return id;
	}

	public void join(int id1, int id2) {
		id1 = parentOf(id1); id2 = parentOf(id2);
		if(id1 == id2)
			return;

		int rank1 = items[id1].rank, rank2 = items[id2].rank;
		if(rank1 < rank2)
			items[id1].ancestor = id2;
		else
			items[id2].ancestor = id1;
		
		if(rank1 == rank2)
			++items[id1].rank;

		--groups;
	}
	
	public String toString() {
		StringBuffer result = new StringBuffer(groups + " groups: ");
		TreeMap<Integer, ArrayList<Integer>> mapping = new TreeMap<Integer, ArrayList<Integer>>();
		for(int i = 0, lim = items.length; i<lim; ++i) {
			int parentId = parentOf(i);
			ArrayList<Integer> members = mapping.get(parentId);
			if(members==null) {
				members = new ArrayList<Integer>();
				mapping.put(parentId, members);
			}
			members.add(i);
		}

		result.append(mapping);

		return result.toString();
	}

	public static void main(String args[]) {
		UnionFind uf = new UnionFind(10);
		System.out.println("Initial uf:" + uf);
		// uf.join(2, 1);
		// uf.join(0, 4);
		// uf.join(0, 5);
		// uf.join(6, 3);
		// uf.join(7, 4);
		// uf.join(9, 4);
		// uf.join(2, 6);
		// uf.join(2, 5);
		// uf.join(1, 8);
		// System.out.println(uf);

		uf.join(0, 3);
		System.out.println(uf);
		uf.join(4, 5);
		System.out.println(uf);
		uf.join(1, 9);
		System.out.println(uf);
		uf.join(2, 8);
		System.out.println(uf);
		uf.join(7, 4);
		System.out.println(uf);
		uf.join(9, 0);
		System.out.println(uf);
		uf.join(7, 8);
		System.out.println(uf);
		uf.join(1, 6);
		System.out.println(uf);
		uf.join(0, 5);
		System.out.println(uf);
	}
}
