/*
    Implementation of the UnionFind data structure described here:
        https://en.wikipedia.org/wiki/Disjoint-set_data_structure

	Tested with Java 8

	Uses '../common/RelevantLines.java', so:

	- compile with:  javac -cp "../common/" UnionFind.java 
		(this will compile 'RelevantLines.java' as well)

	- launch like:   java -cp ".;../common/" UnionFind


    @2017 Florin Tulba (florintulba@yahoo.com)
*/


import java.util.*;
import java.io.*;
//import RelevantLines; // "It is a compile time error to import a type from the unnamed package." (Java language specification)

/**
 * Provides the problem input data
 */
class ScenarioProvider {

	/**
	 * Helps working with element indices of pairs of elements to be joined
	 */
	public class ElemPair {
		protected int idx1;
		protected int idx2;

		public ElemPair(int idx1_, int idx2_) {
			idx1 = idx1_; idx2 = idx2_;
		}

		public int firstIdx() {return idx1;}
		public int secondIdx() {return idx2;}
	}

	static final String scenarioFile = "testScenario.txt"; // name of the scenario file

	RelevantLines parser;				// the provider of the relevant lines from the scenario file
	protected int items = 0;			// number of elements from the described scenario
	protected boolean valid = true;		// becomes false as soon as IO / parsing errors are found in the scenario file

	/**
	  * Reads itemsCount from 'testScenario.txt'
	  */
	public ScenarioProvider() {
        try {
            parser = new RelevantLines(scenarioFile);
        } catch(IOException e) {
        	System.err.println("Couldn't open the scenario file: " + scenarioFile);
        	valid = false;
        	return;
        }

        try {
        	String line = null;
	        while((line = parser.nextLine()) != null) {
		        Scanner s = new Scanner(line);
		        if(s.hasNextInt()) {
		        	items = s.nextInt();
		        	return;
		        } else {
		        	valid = false;
		        	break;
		        }
	        }
        } catch(IOException e) {
        	valid = false;
        }

        if( ! valid)
        	System.err.println("Couldn't read the items count from the scenario file! Please correct this error and then try again!");
	}

	public int itemsCount() { return items; }

	/**
	  * Reads next pair of element indices
	  */
	public ElemPair nextElemPair() {
		if( ! valid) {
			System.out.println("Please correct previous errors before calling ScenarioProvider::nextElemPair()!");
			return null;
		}

        try {
        	String line = null;
	        while((line = parser.nextLine()) != null) {
		        Scanner s = new Scanner(line);
		        int idx1, idx2;
		        if(s.hasNextInt()) {
		        	idx1 = s.nextInt();
			        if(s.hasNextInt()) {
			        	idx2 = s.nextInt();
			        	return new ElemPair(idx1, idx2);
			        }
		        }
		        valid = false;
		        break;
	        }
        } catch(IOException e) {
        	valid = false;
        }

        if( ! valid)
        	System.err.println("Couldn't read the next pair of element indices from the scenario file! Please correct this error and then try again!");

		System.out.println("No other pairs to connect in this scenario!");
		return null;
	}
}

/**
 * The UnionFind class. It's the one in charge of validating the data input
 */
public class UnionFind {
	/**
	  * Data specific for each item of the Union Find
	  */
	protected class UfItem {
		public int ancestor;	// the most distant known ancestor
		public int rank;		// initial depth of a subtree rooted on this item (some children might migrate closer to the root of this item)
		public UfItem(int id) {
			ancestor = id;
			rank = 0;
		}
	}

	protected UfItem[] items;	// the managed items
	protected int groups;			// current count of the groups formed from the items

	/**
	 * Create itemCount items that are initially separated
	 */
	public UnionFind(int itemsCount) {
		groups = itemsCount;
		items = new UfItem[itemsCount];
		for(int i=0; i<itemsCount; ++i)
			items[i] = new UfItem(i);
	
		System.out.println(" Initially: " + toString());
		if(items.length < 2)
			System.out.println("Note that this problem makes sense only for at least 2 elements!");
	}

	/**
	 * Checks if the provided index might be a valid element index
	 */
	protected boolean validateIndex(int idx) {
		if(idx < items.length)
			return true;

		System.err.println("Invalid element index: " + idx);
		return false;
	}

	/**
	 * Find parent operation
	 */
	protected int parentOf(int id) {
		assert(validateIndex(id));

		int parentId;
		while(id != (parentId = items[id].ancestor))
			id = items[id].ancestor = items[parentId].ancestor;
		
		return id;
	}

	/**
	 * Connect id1 and id2
	 */
	public void join(int id1, int id2) {
		System.out.format("%3d - %3d : ", id1, id2);
		if( ! validateIndex(id1) || ! validateIndex(id2))
			return;

		id1 = parentOf(id1); id2 = parentOf(id2);
		if(id1 == id2) {
			System.out.println(toString());
			return;
		}

		int rank1 = items[id1].rank, rank2 = items[id2].rank;
		if(rank1 < rank2)
			items[id1].ancestor = id2;
		else
			items[id2].ancestor = id1;
		
		if(rank1 == rank2)
			++items[id1].rank;

		--groups;
		System.out.println(toString());
		if(groups == 1)
			System.out.println("All elements are now connected!");
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
		ScenarioProvider sp = new ScenarioProvider();
		UnionFind uf = new UnionFind(sp.itemsCount());

		ScenarioProvider.ElemPair elemPair = null;
		while(null != (elemPair = sp.nextElemPair()))
			uf.join(elemPair.firstIdx(), elemPair.secondIdx());
	}
}
