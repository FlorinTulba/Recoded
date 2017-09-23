/*
	Finding the most distant pair of sorted elements in an array.

	Compiled with Java 1.8.

	@2017 Florin Tulba (florintulba@yahoo.com)
*/

import java.util.AbstractList;
import java.util.ArrayList;
import java.util.Arrays;
import java.util.Collections;
import java.util.Iterator;
import java.util.List;
import java.util.ListIterator;
import java.util.NoSuchElementException;
import java.util.SortedMap;
import java.util.TreeMap;

/**
 * Provides successive permutations of an initial array, which modify this array
 */
class Permutator<T> {
    protected ArrayList<T> vals; // the values to be permuted
    protected int startIdx; // where from in vals to start permute the items
    protected int pivot; // absolute index of the last pivot (relative to the
                         // start of vals, not to startIdx)

    protected int valsCount; // count of items within original vals
    protected int items; // count of items to permute
    protected Permutator<T> worker; // executes the permutations of the elements
                                    // positioned from 1 to N

    protected boolean done; // set when all permutations were generated

    protected Permutator(ArrayList<T> vals_, int startIdx_, int pivot_) {
        assert vals_ != null;
        assert pivot_ > startIdx_;
        vals = vals_;
        startIdx = startIdx_;
        pivot = pivot_;
        valsCount = vals.size();
        items = valsCount - startIdx;
        if (items > 0)
            worker = new Permutator<T>(vals, startIdx + 1, startIdx + 2);
        done = false;
    }

    public Permutator(ArrayList<T> vals_) {
        this(vals_, 0, 1);
    }

    protected void swap(int i, int j) {
        assert i < valsCount && j < valsCount;
        final T val = vals.get(i);
        vals.set(i, vals.get(j));
        vals.set(j, val);
    }

    protected void reset() {
        done = false;
        pivot = startIdx + 1;
        if (items > 0)
            worker.reset();
    }

    protected void lshift() {
        final T leftMost = vals.get(startIdx);

        for (int i = startIdx; i < valsCount - 1; ++i)
            vals.set(i, vals.get(i + 1));

        vals.set(valsCount - 1, leftMost);
    }

    public boolean nextPermutation() {
        if (done || items < 2)
            return false;

        if (worker.nextPermutation())
            return true;

        if (pivot < valsCount) {
            swap(startIdx, pivot++);
            worker.reset();
            return true;
        }

        lshift();
        done = true;
        return false;
    }
}

/**
 * Details about key elements observed while traversing the array from the right
 */
class Info {
    private int rightIdx; // the index from the right where the value was found

    public int right() {
        return rightIdx;
    }

    // how many relevant items towards left rely on this value
    public int coverage;

    public Info(int rightIdx_, int coverage_) {
        rightIdx = rightIdx_;
        coverage = coverage_;
    }

    public Info(int rightIdx_) {
        this(rightIdx_, 1);
    }
};

/**
 * Concatenation of the key and Info
 */
class Key_Info<T> {
    protected T _val; // the maximum value key
    public Info info;

    public T val() {
        return _val;
    }

    public Key_Info(T val, Info info_) {
        _val = val;
        info = info_;
    }
}

/**
 * The key elements observed while traversing the array from the right, which
 * are the only options for right pair members.
 */
class RightOptions<T extends Comparable<? super T>> {
    protected SortedMap<T, Info> data; // the key elements with local maximum
                                       // values

    private boolean addAllowed; // prevent adding after calling doneAdding()

    protected void validate() throws NoSuchElementException {
        if (empty())
            throw new NoSuchElementException("No options!");
    }

    public RightOptions() {
        data = new TreeMap<T, Info>();
        addAllowed = true;
    }

    public boolean empty() {
        return data.isEmpty();
    }

    public int size() {
        return data.size();
    }

    // Registers a newly found maximum value
    public void addNew(T val, int right, int coverage)
            throws IllegalStateException {
        if (!addAllowed)
            throw new IllegalStateException(
                    "Cannot add after calling doneAdding()!");
        data.put(val, new Info(right, coverage));
    }

    public void addNew(T val, int right) throws IllegalStateException {
        addNew(val, right, 1);
    }

    // Sets addAllowed on false
    public void doneAdding() {
        addAllowed = false;
    }

    // Last introduced option
    public Key_Info<T> lastKnown() throws NoSuchElementException {
        validate();
        final T lastKnownKey = data.lastKey();
        return new Key_Info<>(lastKnownKey, data.get(lastKnownKey));
    }

    // Returns the first option larger than val. val must be less than the
    // largest option in data
    public Key_Info<T> optionFor(T val) throws NoSuchElementException {
        validate();
        assert val.compareTo(data.lastKey()) < 0;
        SortedMap<T, Info> subMap = data.tailMap(val);
        T rightKey = subMap.firstKey();
        if (rightKey.compareTo(val) == 0) {
            assert subMap.keySet().size() > 1;
            Iterator<T> itKey = subMap.keySet().iterator();
            assert itKey.hasNext();
            itKey.next();
            assert itKey.hasNext();
            rightKey = itKey.next();
        }
        return new Key_Info<>(rightKey, subMap.get(rightKey));
    }

    // Withdraw coverage from latest options until reaching count or until data
    // gets empty
    public void reduce(int count) {
        while (count > 0 && !empty()) {
            final T lastKnownKey = data.lastKey();
            final int coverageOfLast = data.get(lastKnownKey).coverage,
                    removableCoverage = Math.min(count, coverageOfLast);
            if (removableCoverage == coverageOfLast) {
                data.remove(lastKnownKey);
            } else
                data.get(lastKnownKey).coverage -= removableCoverage;
            count -= removableCoverage;
        }
    }

    public void reduce() {
        reduce(1);
    }
};

/**
 * Traverses an array to determine the most distant pair of sorted elements
 */
public class FurthestSortedPair {
    /**
     * Slow, but safe method with O(N^2) for finding the furthest pair of sorted
     * values within vals.
     * 
     * It uses a main loop to consider all possible left members of the pair.
     * There is also an inner loop that checks for corresponding right members
     * of the pair starting from the right end of the array and stopping when
     * the pair spacing is better / inferior than the previous best.
     */
    public static <T extends Comparable<? super T>> int referenceResult(
            AbstractList<T> vals) {
        final int valsCount = vals.size();
        if (0 == valsCount)
            return 0;

        int maxDist = 0;
        Iterator<T> itLeft = vals.iterator();
        for (int left = 0; left + maxDist + 1 < valsCount; ++left) {
            assert itLeft.hasNext();
            final T leftVal = itLeft.next();
            ListIterator<T> itRight = vals.listIterator(valsCount);
            for (int right = valsCount - 1; right > left + maxDist; --right) {
                assert itRight.hasPrevious();
                if (leftVal.compareTo(itRight.previous()) < 0) {
                    maxDist = right - left;
                    break;
                }
            }
        }
        return maxDist;
    }

    // private static final double ONE_OVER_LOG2 = 1. / Math.log(2);

    // Computes (int)ceiling(log2(sz))
    private static int ceilLog2(int sz) {
        assert sz > 0;
        // return (int)Math.ceil(Math.log(sz) * ONE_OVER_LOG2);
        return 32 - Integer.numberOfLeadingZeros(sz); // should be faster
    }

    /**
     * This approach of finding the furthest pair of sorted values within vals
     * works in O(N*log(N)).
     * 
     * 
     * It also uses a main loop to consider all possible left members of the
     * pair. However it considers the following facts:
     * 
     * - larger left members of the pair than previously considered cannot
     * deliver better result. This means skipping left pair members larger than
     * the minimum left pair member previously analyzed
     * 
     * - similarly, for a given left pair member, the corresponding right pair
     * member might be strictly one of the (updated) maximum values found while
     * traversing the array from right towards left
     * 
     * - the analysis of the right pair member for the first possible left pair
     * member (the very first loop) allows improving the next passes. So, the
     * index of each newly encountered maximum value (right to left traversal)
     * can be recorded.
     * 
     * - the search for the first larger value than left pair member within the
     * previously mentioned array of stored maximum values (recorded in
     * ascending order) can be performed using a binary search, obtaining a
     * log(N) for the inner loop
     *
     * @param vals
     *            the array of elements
     * @param comparesCount
     *            array of 1 element with the overestimated value of the number
     *            of required compare operations
     * 
     * @return the distance between the furthest pair of sorted values of the
     *         array
     */
    public static <T extends Comparable<? super T>> int improvedMethod(
            AbstractList<T> vals, int[] comparesCount) {
        final int valuesCount = vals.size();
        comparesCount[0] = 0;

        if (valuesCount < 2)
            return 0; // There's no pair of elements yet

        final int lastIdx = valuesCount - 1;
        Iterator<T> itLeft = vals.iterator();
        ListIterator<T> itRight = vals.listIterator(valuesCount);
        assert itLeft.hasNext();
        assert itRight.hasPrevious();
        T leftVal = itLeft.next(), rightVal = itRight.previous();
        ++comparesCount[0];
        if (leftVal.compareTo(rightVal) < 0)
            return lastIdx;

        int maxDist = 0, comparesCount_ = comparesCount[0];

        // First inspection of the array considering first element as the left
        // pair member
        int left = 0;
        RightOptions<T> rightOptions = new RightOptions<T>();
        rightOptions.addNew(rightVal, lastIdx, 0); // let coverage field be set
                                                   // to 1 within the loop
        for (int right = lastIdx; right > /* left + */maxDist; --right, rightVal = itRight
                .hasPrevious() ? itRight.previous() : null) {
            assert rightVal != null;
            ++comparesCount_;
            if (rightVal.compareTo(rightOptions.lastKnown().val()) > 0) {
                // new maximum, which can be then compared against leftVal
                ++comparesCount_;
                if (leftVal.compareTo(rightVal) < 0) {
                    maxDist = right/* - left */;
                    break;
                }

                // add the new max only if this matters when left=1
                rightOptions.addNew(rightVal, right);

            } else
                ++rightOptions.lastKnown().info.coverage;
        }
        rightOptions.doneAdding();

        // Checking all remaining potential left pair members
        T minVal = leftVal; // init min
        for (left = 1; left + maxDist < lastIdx; ++left) {
            assert itLeft.hasNext();
            leftVal = itLeft.next();

            rightOptions.reduce(); // a value should be popped out for each
                                   // iteration

            // assessing only local minimum left pair members
            ++comparesCount_;
            if (leftVal.compareTo(minVal) >= 0)
                continue;

            minVal = leftVal; // renew min

            // Check if there is no right pair member
            assert !rightOptions.empty();
            ++comparesCount_;
            if (leftVal.compareTo(rightOptions.lastKnown().val()) >= 0)
                continue;

            // The appropriate right pair member can be found using binary
            // search
            comparesCount_ += ceilLog2(rightOptions.size()); // overestimate of
                                                             // the compare
                                                             // ops performed
            final Key_Info<T> rightInfo = rightOptions.optionFor(leftVal);

            // Discarding (rightInfo.info.right() - left) - maxDist elements
            // from rightOptions (if there are so many)
            // and assigning rightInfo.info.right() - left to maxDist
            final int newMaxDist = rightInfo.info.right() - left;
            rightOptions.reduce(newMaxDist - maxDist);
            maxDist = newMaxDist;
        }

        // At this point, rightOptions should be empty, or contain one option
        // with coverage 1
        assert rightOptions.empty() || (rightOptions.size() == 1
                && rightOptions.lastKnown().info.coverage == 1);

        comparesCount[0] = comparesCount_;
        return maxDist;
    }

    // Compares the results of the 2 approaches and reports errors and compare
    // operations count
    protected static <T extends Comparable<? super T>> int checkUseCase(
            AbstractList<T> vals, int[] errorsCount, boolean verbose) {
        final int[] comparesCount = new int[] { 0 };
        final int refRes = referenceResult(vals),
                res = improvedMethod(vals, comparesCount);
        if (refRes != res) {
            ++errorsCount[0];
            System.err.println("For the array below, the expected result was "
                    + refRes + ", but obtained " + res + " instead.");
            for (final T val : vals)
                System.err.print(val + ", ");
            System.err.println("\b\b \n");
        } else if (verbose) {
            System.out.println(
                    "Furthest sorted pair of elements is at a distance of "
                            + res + " in the array from below. It needed "
                            + comparesCount[0] + " compare ops.");
            for (final T val : vals)
                System.out.print(val + ", ");
            System.out.println("\b\b \n");
        }

        return comparesCount[0];
    }

    protected static <T extends Comparable<? super T>> int checkUseCase(
            AbstractList<T> vals, int[] errorsCount) {
        return checkUseCase(vals, errorsCount, false);
    }

    private static final int TIMES = 1000, // count of random arrays to be
                                           // checked
            VALUES_COUNT = 1000; // size of each random array

    public static void main(String[] args) {
        final int[] errorsCount = new int[] { 0 };

        // The container can be derived from AbstractList and may contain any T
        // that is Comparable
        ArrayList<Integer> vals = new ArrayList<Integer>();

        // Empty array case
        checkUseCase(vals, errorsCount);

        // Single element array case
        vals.add(100);
        checkUseCase(vals, errorsCount);

        // Sorted array case
        vals.clear();
        vals.ensureCapacity(VALUES_COUNT);
        for (int i = 0; i < VALUES_COUNT; ++i)
            vals.add(i);
        checkUseCase(vals, errorsCount, true);

        // Descending sorted array case
        vals.clear();
        vals.ensureCapacity(VALUES_COUNT);
        for (int i = VALUES_COUNT - 1; i >= 0; --i)
            vals.add(i);
        checkUseCase(vals, errorsCount, true);

        // Random arrays cases
        System.out.println("Checking random arrays ...");
        Collections.shuffle(vals);
        checkUseCase(vals, errorsCount, true);

        for (int t = 0; t < TIMES; ++t) {
            Collections.shuffle(vals);
            checkUseCase(vals, errorsCount);
        }

        // Searching for the worst case scenario within the 3628800 permutations
        // of an array of 10 elements
        System.out.println("Looking for the worst case scenario ...");
        int maxComparesCount = 0;
        List<Integer> worstConfig = null;
        vals.clear();
        vals.ensureCapacity(10);
        for (int i = 1; i <= 10; ++i)
            vals.add(i);

        Permutator<Integer> permutator = new Permutator<Integer>(vals);
        do {
            final int comparesCount = checkUseCase(vals, errorsCount);
            if (comparesCount > maxComparesCount) {
                maxComparesCount = comparesCount;
                worstConfig = new ArrayList<Integer>(vals);
            }
        } while (permutator.nextPermutation());
        System.out.println("The worst configuration from below produced "
                + maxComparesCount + " compare ops.");
        for (final Integer val : worstConfig)
            System.out.print(val + ", ");
        System.out.println("\b\b \n");

        // Inspecting the worst case scenario found above (depends on the order
        // of the permutations):
        vals = new ArrayList<>(Arrays.asList(10, 8, 9, 7, 6, 5, 4, 3, 2, 1));
        final int[] comparesCount = new int[] { 0 };
        improvedMethod(vals, comparesCount);
        System.out.println("The worst case scenario required "
                + comparesCount[0] + " compare ops.");

        if (errorsCount[0] > 0)
            System.err.println("There were " + errorsCount[0] + " errors!");
        else
            System.out.println("There were no errors");
    }
}
