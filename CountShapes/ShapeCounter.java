/*
	Counting all possible triangles and (convex) quadrilaterals from geometric figures traversed by a number of lines.
	See the tested figures in 'TestFigures/'.

	Compiled with Java 1.8.

	Uses '../common/RelevantLines.java', so:

	- compile with:  javac -cp "../common/" ShapeCounter.java 
		(this will compile 'RelevantLines.java' as well)

	- launch like:   java -cp ".;../common/" ShapeCounter


	@2017 Florin Tulba (florintulba@yahoo.com)
*/

import java.util.Scanner;
import java.util.ArrayList;
import java.util.BitSet;
import java.util.TreeMap;
import java.io.IOException;
//import RelevantLines; // "It is a compile time error to import a type from the unnamed package." (Java language specification)

/**
 * Counts triangles and convex quadrilaterals from a figure
 */
public class ShapeCounter {
	protected int N = 0;	// Points Count
	protected int L = 0;	// Lines Count

	protected int triangles_ = 0;	// Count of triangles
	protected int convQuadr = 0;	// Count of convex quadrilaterals

	ArrayList<String> pointNames = new ArrayList<>(); // the names of the points
	ArrayList<BitSet> lineMembers; // the indices of the members of each line
	ArrayList<BitSet> connections = new ArrayList<>(); // points connectivity matrix
	ArrayList<BitSet> membership = new ArrayList<>(); // membership of points to the lines

	// for each point a map between lineIdx and rankWithinLine
	ArrayList<TreeMap<Integer, Integer>> membershipAsRanks = new ArrayList<>();

	/**
	 * Configures the ShapeCounter based on the sequences of named points from the lines from the figure.
	 * Prepares the entire infrastructure needed while counting the shapes.
	 */
	public ShapeCounter(ArrayList<ArrayList<String>> lines) {
		lineMembers = new ArrayList<>(L = lines.size());
		TreeMap<String, Integer> pointsIndices = new TreeMap<>();
		for (int lineIdx = 0; lineIdx < L; ++lineIdx) {
			ArrayList<String> line = lines.get(lineIdx);
			int pointsOnLine = line.size();
			ArrayList<Integer> memberIndices = new ArrayList<>(pointsOnLine);
			BitSet memberIndicesBitset = new BitSet(N);
			for (int pointRank = 0; pointRank < pointsOnLine; ++pointRank) {
				String pointName = line.get(pointRank);
				int pointIdx;
				if (!pointsIndices.containsKey(pointName)) {
					pointNames.add(pointName);
					for (BitSet prevMembers : lineMembers)
						prevMembers.clear(N);
					for (BitSet conns : connections)
						conns.clear(N);
					memberIndicesBitset.set(N);
					pointsIndices.put(pointName, pointIdx = N++);
					connections.add(new BitSet(N));
					membership.add(new BitSet(L));
					membershipAsRanks.add(new TreeMap<>());
				} else {
					pointIdx = pointsIndices.get(pointName);
					memberIndicesBitset.set(pointIdx);
				}
				BitSet lastPointConns = connections.get(pointIdx);
				for (Integer prevIdx : memberIndices) {
					connections.get(prevIdx).set(pointIdx);
					lastPointConns.set(prevIdx);
				}
				memberIndices.add(pointIdx);
				membership.get(pointIdx).set(lineIdx);
				membershipAsRanks.get(pointIdx).put(lineIdx, pointRank);
			}
			lineMembers.add(memberIndicesBitset);
		}

		/*
		// Uncomment to verify the correctness of the loaded scenario
		for (int i = 0; i < N; ++i) {
			System.out.print(pointNames.get(i) + ": connections " +
							connections.get(i) + " ; member of lines " +
							membership.get(i) + " ; pos in lines {");
			TreeMap<Integer, Integer> lineMembership = membershipAsRanks.get(i);
			for (Integer lineIdx : lineMembership.keySet())
				System.out.print(lineMembership.get(lineIdx) + "(l" + lineIdx + ") ");
			System.out.println("\b}");
		}
		
		for (int i = 0; i < L; ++i)
			System.out.println("L" + i + ": members {" + lineMembers.get(i) + '}');
		
		System.out.println();
		*/
	}

	/**
	 * Parameter type for the methods below providing the indices of 2 points crossed by a line
	 */
	protected class Line {
		public int first, second; // the indices of the 2 points defining the line

		public Line(int firstPoint, int secondPoint) {
			first = firstPoint;
			second = secondPoint;
		}
	}

	/**
	 * @param l1 one line from a potential quadrilateral
	 * @param l2 the line across from l1 in the potential quadrilateral
	 * @param memL1 'and'-ed memberships (which lines include each point) of the 2 points from l1
	 * @param memL2_1 membership (which lines include the point) of one point from l2
	 * @param memL2_2 membership (which lines include the point) of the other point from l2
	 *
	 * @return true only if the extended lines l1 and l2 don't intersect, or intersect strictly outside the shape described by the 4 points from l1 and l2
	 */
	protected boolean allowedIntersection(Line l1, Line l2, BitSet memL1, BitSet memL2_1, BitSet memL2_2) {
		int lineIdxPair1 = memL1.nextSetBit(0);
		if (memL2_1.get(lineIdxPair1) || memL2_2.get(lineIdxPair1))
			return false; // one of the provided points from L2 are inside L1

		BitSet tempBitSet = (BitSet) memL2_1.clone();
		tempBitSet.and(memL2_2);
		int lineIdxPair2 = tempBitSet.nextSetBit(0);
		tempBitSet = (BitSet) lineMembers.get(lineIdxPair1).clone();
		tempBitSet.and(lineMembers.get(lineIdxPair2));
		int intersectionPoint = tempBitSet.nextSetBit(0);
		if (intersectionPoint != -1) {
			int rank1 = membershipAsRanks.get(l1.first).get(lineIdxPair1),
					rank2 = membershipAsRanks.get(l1.second).get(lineIdxPair1);
			if (rank1 > rank2) {
				int temp = rank1;
				rank1 = rank2;
				rank2 = temp;
			}
			TreeMap<Integer, Integer> intersectionPointMembership = membershipAsRanks.get(intersectionPoint);
			int rank = intersectionPointMembership.get(lineIdxPair1);
			if (rank1 <= rank && rank <= rank2)
				return false;

			rank1 = membershipAsRanks.get(l2.first).get(lineIdxPair2);
			rank2 = membershipAsRanks.get(l2.second).get(lineIdxPair2);
			if (rank1 > rank2) {
				int temp = rank1;
				rank1 = rank2;
				rank2 = temp;
			}
			rank = intersectionPointMembership.get(lineIdxPair2);
			if (rank1 <= rank && rank <= rank2)
				return false;
		}

		return true;
	}

	/**
	 * Checks convexity of p1-p4 quadrilateral, based on the membership of each point to the available lines
	 */
	protected boolean convex(int p1, BitSet mem1, int p2, BitSet mem2, int p3, BitSet mem3, int p4, BitSet mem4) {
		assert (Math.max(p1, p2) < N && Math.max(p3, p4) < N);
		assert (p1 != p2 && p1 != p3 && p1 != p4 && p2 != p3 && p2 != p4 && p3 != p4);

		// Extended p1-p2 and p3-p4 shouldn't touch
		BitSet tempBitSet = (BitSet) mem1.clone();
		tempBitSet.and(mem2);
		if (!allowedIntersection(new Line(p1, p2), new Line(p3, p4), tempBitSet, mem3, mem4))
			return false;

		// Extended p2-p3 and p4-p1 shouldn't touch
		tempBitSet = (BitSet) mem2.clone();
		tempBitSet.and(mem3);
		if (!allowedIntersection(new Line(p2, p3), new Line(p4, p1), tempBitSet, mem4, mem1))
			return false;

		return true;
	}

	/**
	 * Performs the actual shape counting
	 */
	public void process() {
		BitSet maskP1 = new BitSet(N);
		maskP1.set(0, N);
		for (int p1 = 0, limP1 = N - 2; p1 < limP1; ++p1) {
			BitSet mem1 = membership.get(p1);
			maskP1.clear(p1); // Ignore connections before and including P1
			BitSet connOfP1Bitset = (BitSet) connections.get(p1).clone();
			connOfP1Bitset.and(maskP1);
			int countConnOfP1 = connOfP1Bitset.cardinality();
			if (countConnOfP1 < 2)
				continue; // Triangles require 2 connected points to P1. If they are not available, check next available P1

			ArrayList<Integer> connOfP1 = new ArrayList<>(countConnOfP1);
			for (int p = connOfP1Bitset.nextSetBit(0), idx = 0; idx < countConnOfP1; p = connOfP1Bitset
					.nextSetBit(p + 1), ++idx)
				connOfP1.add(p);

			for (int idxP2 = 0, p2 = connOfP1.get(0), limP2 = countConnOfP1 - 1; idxP2 < limP2; p2 = connOfP1
					.get(++idxP2)) {
				BitSet mem2 = membership.get(p2);
				BitSet mem1and2 = (BitSet) mem1.clone();
				mem1and2.and(mem2);
				for (int idxLastP = idxP2 + 1; idxLastP < countConnOfP1; ++idxLastP) {
					int lastP = connOfP1.get(idxLastP);
					BitSet memLast = membership.get(lastP);
					BitSet tempBitSet = (BitSet) mem1and2.clone();
					tempBitSet.and(memLast);
					if (!tempBitSet.isEmpty()) // coll(p1, p2, lastP)
						continue; // Ignore collinear points

					if (connections.get(p2).get(lastP)) {
						++triangles_;
						System.out.print("<" + pointNames.get(p1) + pointNames.get(p2) + pointNames.get(lastP) + "> ");
					}

					BitSet connOfP2_LastP_Bitset = (BitSet) connections.get(p2).clone();
					connOfP2_LastP_Bitset.and(connections.get(lastP));
					connOfP2_LastP_Bitset.and(maskP1);
					BitSet mem1and2or2andLast = (BitSet) mem2.clone();
					mem1and2or2andLast.and(memLast);
					mem1and2or2andLast.or(mem1and2);
					for (int p3 = connOfP2_LastP_Bitset.nextSetBit(0); p3 != -1; p3 = connOfP2_LastP_Bitset
							.nextSetBit(p3 + 1)) {
						BitSet mem3 = membership.get(p3);
						tempBitSet = (BitSet) mem1and2or2andLast.clone();
						tempBitSet.and(mem3);
						if (!tempBitSet.isEmpty()) // coll(p1, p2, p3) ||
													// coll(lastP, p2, p3)
							continue; // Ignore collinear points

						if (convex(p1, mem1, p2, mem2, p3, mem3, lastP, memLast)) {
							++convQuadr;
							System.out.print("[" + pointNames.get(p1) + pointNames.get(p2) + pointNames.get(p3)
									+ pointNames.get(lastP) + "] ");
						}
					}
				}
			}
		}
		System.out.println();
	}

	public int triangles() {
		return triangles_;
	}

	public int convexQuadrilaterals() {
		return convQuadr;
	}

	public static void main(String[] args) {
		// Select one of the following scenario files (uncommenting it and commenting the others)
		// or create a new scenario and open your own file:

		// Scenario from figure 'count6Shapes.png'
		String testFileName = "TestFigures/TextVersions/count6shapes.txt";		// Manually generated (same labels as in the corresponding figure)
		//String testFileName = "TestFigures/TextVersions/count6shapes.png.txt";	// Generated by the figure interpreter written in Matlab/Octave

		// Scenario from figure 'count9Shapes.png'
		//String testFileName = "TestFigures/TextVersions/count9shapes.txt";		// Manually generated (same labels as in the corresponding figure)
		//String testFileName = "TestFigures/TextVersions/count9shapes.png.txt";	// Generated by the figure interpreter written in Matlab/Octave

		// Scenario from figure 'count100Shapes.png'
		//String testFileName = "TestFigures/TextVersions/count100shapes.txt";	// Manually generated (same labels as in the corresponding figure)
		//String testFileName = "TestFigures/TextVersions/count100shapes.png.txt";// Generated by the figure interpreter written in Matlab/Octave

		// Scenario from figure 'count673Shapes.png'
		//String testFileName = "TestFigures/TextVersions/count673shapes.txt";	// Manually generated (same labels as in the corresponding figure)
		//String testFileName = "TestFigures/TextVersions/count673shapes.png.txt";// Generated by the figure interpreter written in Matlab/Octave

		// Scenario from figure 'count25651Shapes.png'
		//String testFileName = "TestFigures/TextVersions/count25651shapes.png.txt";// Generated by the figure interpreter written in Matlab/Octave

		ArrayList<ArrayList<String>> lines = new ArrayList<>();
		try {
			RelevantLines parser = new RelevantLines(testFileName);
			String line = null;
			while((line = parser.nextLine()) != null) {
				ArrayList<String> linePoints = new ArrayList<>();
		        Scanner s = new Scanner(line);
		        while(s.hasNext())
		        	linePoints.add(s.next());
		        lines.add(linePoints);
			}
		} catch(IOException e) {
			System.err.println("Couldn't open or read from " + testFileName);
			return;
		}

		ShapeCounter sc = new ShapeCounter(lines);
		sc.process();
		int totalShapes = sc.triangles() + sc.convexQuadrilaterals();
		System.out.println("There are " + sc.triangles() + " triangles and " + sc.convexQuadrilaterals()
				+ " convex quadrilaterals, which means " + totalShapes + " convex shapes in total.");
	}
}
