
/**
 * Given a matrix, for any element with value 0,
 * set all elements from the corresponding row and column on 0, as well.
 * 
 * Multithreading implementation for Java 1.8.
 * 
 * @2017 Florin Tulba (florintulba@yahoo.com)
 * 
 */

import java.util.ArrayList;
import java.util.Arrays;
import java.util.BitSet;
import java.util.Random;

/**
 * A pair of values: the start index of a range and the past last index
 */
class Range {
	private int startIdx;
	private int pastLastIdx;

	public Range(int startIdx_, int pastLastIdx_) {
		startIdx = startIdx_;
		pastLastIdx = pastLastIdx_;
	}

	public int start() {
		return startIdx;
	}

	public int pastLast() {
		return pastLastIdx;
	}
}

/**
 * Implements the algorithm to expand the zeros and provides code for testing
 * it.
 */
public class ExpandZeros {
	protected static final int processorsCount = Runtime.getRuntime().availableProcessors();

	// generator of random values
	protected static final Random randGen = new Random(System.nanoTime());

	protected static final int TIMES = 1000; // iterations count

	// Max matrix dimensions
	protected static final int mMax = 500; // max number of rows
	protected static final int nMax = 500; // max number of columns
	protected static final int dimAMax = mMax * nMax; // max number of elements
	protected static final short origZerosPercentage = 4; // Desired percentage
															// of zeros within
															// generated
															// matrices

	protected int[] a; // the linearized matrix, preallocated for max allowed
						// size
	protected int m; // the actual number of rows
	protected int n; // the actual number of columns
	protected int dim; // actual number of matrix elements

	protected BitSet checkRows; // the correct rows containing value 0
	protected BitSet checkCols; // the correct columns containing value 0

	protected BitSet foundRows; // the found rows containing value 0
	protected BitSet foundCols; // the found columns containing value 0

	/**
	 * It only preallocates the matrix for max allowed size
	 */
	public ExpandZeros() {
		a = new int[dimAMax];

		m = n = dim = 0;
		checkCols = checkRows = foundCols = foundRows = null;
	}

	/**
	 * Chooses the dimensions and the elements of matrix a and provides
	 * checkRows and checkCols
	 */
	protected void randInitMat() {
		m = randGen.nextInt(mMax - 15) + 15; // random between [15, mMax)
		n = randGen.nextInt(nMax - 15) + 15; // random between [15, nMax)
		dim = m * n;
		checkRows = new BitSet(m);
		checkCols = new BitSet(n);

		// Set all elements first on non-zero values
		for (int i = 0; i < dim; ++i)
			a[i] = randGen.nextInt(999) + 1; // random between [1, 1000)

		// Choose some coordinates to be set to 0
		final int zerosCount = dim * origZerosPercentage / 100;
		for (int i = 0; i < zerosCount; ++i) {
			final int pos = randGen.nextInt(dim), row = pos / n, col = pos % n;
			a[pos] = 0;
			checkRows.set(row);
			checkCols.set(col);
		}
	}

	/**
	 * Signals any mismatches between (checkCols - foundCols) and (checkRows -
	 * foundRows)
	 * 
	 * @return true if there are no mismatches
	 */
	protected boolean correct() {
		if (null == checkRows || null == checkCols || null == foundRows || null == foundCols)
			return true;

		if (!checkRows.equals(foundRows)) {
			System.err.println("Rows Mismatches:");
			System.err.println("\tExpected: " + checkRows);
			System.err.println("\tReceived: " + foundRows);
			return false;
		}

		if (!checkCols.equals(foundCols)) {
			System.err.println("Columns Mismatches:");
			System.err.println("\tExpected: " + checkCols);
			System.err.println("\tReceived: " + foundCols);
			return false;
		}

		return true;
	}

	/**
	 * Traverses `foundCols`and merges all consecutive columns containing true.
	 * So it creates a set of ranges as explained for type `Range`
	 */
	protected ArrayList<Range> buildColRanges() {
		assert foundCols != null;

		ArrayList<Range> result = new ArrayList<>();
		for (int c = 0; c < n; ++c) {
			if (foundCols.get(c)) {
				// Find how many consecutive columns (after c) need to be set on
				// 0
				int c1 = c + 1;
				for (; c1 < n && foundCols.get(c1); ++c1)
					;
				result.add(new Range(c, c1));
				c = c1;
			}
		}
		return result;
	}

	/**
	 * Detects the rows and columns containing the value 0 within the provided
	 * chunk of data
	 */
	protected void findZerosInChunk(int startRow, int pastLastRow) {
		// Avoids false sharing of foundCols through a local vector
		BitSet localFoundCols = new BitSet(n);

		for (int r = startRow; r < pastLastRow; ++r) {
			boolean rowContains0 = false;
			int idx = r * n;
			for (int c = 0; c < n; ++c) {
				if (a[idx++] == 0) {
					localFoundCols.set(c);
					rowContains0 = true;
				}
			}
			if (rowContains0)
				foundRows.set(r); // setting this outside the inner loop
									// minimizes false sharing of foundRows
		}

		// Perform an union of all localFoundCols into foundCols
		for (int c = 0; c < n; ++c) {
			if (localFoundCols.get(c) && !foundCols.get(c)) {
				// Minimizes false sharing of foundCols by reducing writing
				// operations.
				// Overwriting events actually don't change the data,
				// but they invalidate the corresponding L1 cache line from
				// other threads.
				// This is however cheaper than a synchronization mechanism.
				foundCols.set(c);
			}
		}
	}

	/**
	 * Thread detecting the rows and columns containing the value 0 within the
	 * provided chunk of data
	 */
	protected class ZerosDetector extends Thread {
		protected int startRow; // first row to analyze
		protected int pastLastRow; // the index after the last row to analyze

		public ZerosDetector(int startRow_, int pastLastRow_) {
			startRow = startRow_;
			pastLastRow = pastLastRow_;
		}

		@Override
		public void run() {
			findZerosInChunk(startRow, pastLastRow);
		}
	}

	/**
	 * Expands the zeros from the rows and columns containing an original 0 in
	 * the provided chunk of data
	 */
	protected void expandZerosInChunk(ArrayList<Range> colRanges, int startRow, int pastLastRow) {
		for (int r = startRow; r < pastLastRow; ++r) {
			final int rowStart = r * n;

			// Not using the merge of consecutive rows containing value 0
			// since the merge might cover rows tackled by a different thread
			if (foundRows.get(r)) {
				Arrays.fill(a, rowStart, rowStart + n, 0);
			} else {
				for (Range colRange : colRanges) {
					final int startCol = rowStart + colRange.start();
					final int pastLastCol = rowStart + colRange.pastLast();
					Arrays.fill(a, startCol, pastLastCol, 0);
				}
			}
		}
	}

	/**
	 * Thread expanding the zeros from the rows and columns containing an
	 * original 0 in the provided chunk of data
	 */
	protected class ZerosExpander extends Thread {
		protected ArrayList<Range> colRanges; // which columns need to be set on
												// 0 on each row
		protected int startRow; // first row to analyze
		protected int pastLastRow; // the index after the last row to analyze

		public ZerosExpander(ArrayList<Range> colRanges_, int startRow_, int pastLastRow_) {
			colRanges = colRanges_;
			startRow = startRow_;
			pastLastRow = pastLastRow_;
		}

		@Override
		public void run() {
			expandZerosInChunk(colRanges, startRow, pastLastRow);
		}
	}

	/**
	 * Expands all the zeros from the matrix on the corresponding rows and
	 * columns
	 */
	public void apply() {
		assert null != a && m > 0 && n > 0 && dim == m * n && checkRows != null && checkCols != null;

		foundRows = new BitSet(m);
		foundCols = new BitSet(n);

		ArrayList<Thread> threads = new ArrayList<>();

		// Distribute several consecutive rows to every spawn thread and keep
		// the remaining ones to be processed in this main thread.

		// There is no point spawning a thread unless it has at least 300000
		// elements to analyze
		// The main thread should process most of the time fewer elements, as it
		// also needs to wait for the created threads
		int chunkSz = (int) Math.ceil(Math.max((double) m / processorsCount, 300000. / n));

		int chunkStartRow = 0, chunkPastLastRow = chunkSz;
		int additionalThreads = (int) Math.ceil(m / chunkSz) - 1;
		while (additionalThreads-- > 0) {
			threads.add(new ZerosDetector(chunkStartRow, chunkPastLastRow));
			chunkStartRow = chunkPastLastRow;
			chunkPastLastRow += chunkSz;
		}

		for (Thread t : threads)
			t.start();

		findZerosInChunk(chunkStartRow, m);

		for (Thread t : threads)
			for (;;)
				try {
					t.join();
					break; // join next thread
				} catch (InterruptedException e1) {
					continue; // just retry joining t
				}

		final ArrayList<Range> colRanges = buildColRanges();

		chunkStartRow = 0;
		chunkPastLastRow = chunkSz;
		additionalThreads = threads.size();
		while (additionalThreads-- > 0) {
			threads.set(additionalThreads, new ZerosExpander(colRanges, chunkStartRow, chunkPastLastRow));
			chunkStartRow = chunkPastLastRow;
			chunkPastLastRow += chunkSz;
		}

		for (Thread t : threads)
			t.start();

		expandZerosInChunk(colRanges, chunkStartRow, m);

		for (Thread t : threads)
			for (;;)
				try {
					t.join();
					break; // join next thread
				} catch (InterruptedException e1) {
					continue; // just retry joining t
				}
	}

	/**
	 * Launches the tests for this class
	 * 
	 * @param args
	 *            unused for now
	 */
	public static void main(String[] args) {
		ExpandZeros proc = new ExpandZeros();
		long elapsed = 0;
		long lastStart = 0;
		long totalElems = 0;
		for (int i = 0; i < TIMES; ++i) {
			proc.randInitMat();
			totalElems += proc.dim;
			lastStart = System.nanoTime();
			proc.apply();
			elapsed += System.nanoTime() - lastStart;
			assert proc.correct();
		}

		System.out.println("Processed " + totalElems + " elements in " + TIMES + " matrices in " + elapsed / 1e9 + "s");
		System.out.println("That is " + elapsed / (double) totalElems + " ns / element.");
	}
}
