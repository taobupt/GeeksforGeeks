package array;

/**
 * Created by tao on 1/9/17.
 */

import com.sun.org.apache.xpath.internal.SourceTree;
import leetcodeContest.ContestQuestion;
import org.junit.Before;
import org.junit.Test;

import java.beans.Transient;
import java.util.*;

import static org.junit.Assert.*;

public class ArrayTest {
    ArrayAlgoQuestion arrayAlgoQuestion;

    @Before
    public void setUp() throws Exception {
        arrayAlgoQuestion = new ArrayAlgoQuestion();
    }

    @Test
    public void testMajorityElem() {
        int[] nums = {1, 2, 2, 3, 2, 2, 3};
        System.out.println(arrayAlgoQuestion.majorityElement(nums));
    }

    @Test
    public void testMoveCharacter() {
        String s = "abcdGHiGhjjhGhXiH";
        System.out.println(arrayAlgoQuestion.moveCharacterBetter(s));
    }

    @Test
    public void testThirdMax() {
        int[] nums = {1, 2};
        System.out.println(arrayAlgoQuestion.thirdMax(nums));
    }

    @Test
    public void testQuickSort() {
        int[] nums = {-4, -3, -2, 2, 1, -1, 6, -6, -7, 7, 9, -9};
        arrayAlgoQuestion.quickSort(nums);
    }

    @Test
    public void testGenerate() {
        arrayAlgoQuestion.getRow(3);
    }

    @Test
    public void testFinddisappearedNumber() {
        int[] nums = {4, 3, 2, 7, 8, 2, 3, 1};
        arrayAlgoQuestion.findDisappearNumbersConcise(nums);
    }

    @Test
    public void testMaximumProduct() {
        int[] nums = {0, 2};
        assertEquals(arrayAlgoQuestion.maxProduct(nums), 2);
    }

    @Test
    public void testWordDistance() {
        String[] nums = {"practice", "makes", "perfect", "coding", "makes"};
        WordDistance a = new WordDistance(nums);
        System.out.println(a.shortest("practice", "coding"));

    }

    @Test
    public void testSummaryRanges() {
        int[] nums = {0, 1, 3, 4, 5, 7};
        List<String> res = arrayAlgoQuestion.summaryRanges(nums);
        for (String str : res)
            System.out.println(str);
    }

    @Test
    public void testMissingNumber() {
        int[] nums = {1, 2, 3};
        System.out.println(arrayAlgoQuestion.missingNumberBinarySearch(nums));
    }

    @Test
    public void testSpiralOrder() {
        int[][] nums = {{1, 2, 3}, {4, 5, 6}, {7, 8, 9}};
        List<Integer> res = arrayAlgoQuestion.spiralOrder(nums);
        for (int x : res)
            System.out.println(x);
    }

    @Test
    public void testFindMin() {
        int[] nums = {2, 4, 5, 6, 7, 0, 1};
        System.out.println(arrayAlgoQuestion.findMin(nums));
    }

    @Test
    public void testSearchInRotatedArray() {
        int[] nums = {4, 5, 6, 7, 0, 1, 2};
        assertEquals(arrayAlgoQuestion.searchInSortedArray(nums, 6), 2);
    }

    @Test
    public void testSearchInMatrix() {
        int[][] nums = {{1}};
        assertEquals(arrayAlgoQuestion.searchMatrix(nums, 1), true);
    }

    @Test
    public void testSubset() {
        int[] nums = {1, 2, 3};
        arrayAlgoQuestion.subsetsManipulation(nums);
    }

    @Test
    public void testnextPeremutation() {
        int[] nums = {1, 5, 1};
        arrayAlgoQuestion.nextPermutation(nums);
        for (int x : nums)
            System.out.println(x);
    }

    @Test
    public void testCanJump() {
        int[] nums = {1, 1, 0, 1};
        arrayAlgoQuestion.canJump(nums);
    }

    @Test
    public void testSearchRange() {
        int[] nums = {5, 7, 7, 8, 8, 10};

        int[] res = arrayAlgoQuestion.searchRangeOneBinarySearch(nums, 10);
        for (int x : res)
            System.out.println(x);
    }

    @Test
    public void testGameofLife() {
        int[][] matrix = {{1, 1, 1}, {0, 1, 0}, {1, 0, 1}};
        arrayAlgoQuestion.gameOfLife(matrix);
        int[][] b = matrix.clone();
        matrix[0][0] = 1;
    }

    @Test
    public void InsertAndDelete() {
        RandomizedSet rs = new RandomizedSet();
        System.out.println(rs.remove(0));
        System.out.println(rs.remove(0));
        System.out.println(rs.insert(0));
        System.out.println(rs.getRandom());
        System.out.println(rs.remove(0));
        System.out.println(rs.insert(0));
    }

    @Test
    public void testMerge() {
        List<Interval> intervals = new ArrayList<Interval>();
        arrayAlgoQuestion.merge(intervals);
    }

    @Test
    public void testfirstMissingPositive() {
        int[] nums = {0, 2, 2, 1, 1};
        int x = arrayAlgoQuestion.firstMissingPositiveSaveSpace(nums);
        System.out.println(x);
    }

    @Test
    public void testfindkth() {
        int[] nums1 = {1, 2, 3, 4, 5};
        int[] nums2 = {2, 3, 4, 5, 5, 6};
        int m = nums1.length, n = m + nums2.length;
        for (int i = 1; i <= n; ++i) {
            System.out.println(arrayAlgoQuestion.findkth(nums1, 0, nums2, 0, i));
        }
    }

    @Test
    public void testWordLadder() {
        Set<String> sets = new HashSet<>();
        sets.add("hit");
        sets.add("cog");
        System.out.println(arrayAlgoQuestion.ladderLength("hit", "cog", sets));
    }

    @Test
    public void testSummary() {
        SummaryRanges sr = new SummaryRanges();
        sr.addNum(1);
        List<Interval> res = sr.getIntervals();
        for (Interval val : res)
            System.out.print(val + " ");
        System.out.println();
        sr.addNum(3);
        res = sr.getIntervals();
        for (Interval val : res)
            System.out.print(val + " ");
        System.out.println();
        sr.addNum(7);
        res = sr.getIntervals();
        for (Interval val : res)
            System.out.print(val + " ");
        System.out.println();
        sr.addNum(2);
        res = sr.getIntervals();
        for (Interval val : res)
            System.out.print(val + " ");
        System.out.println();
        sr.addNum(6);
        res = sr.getIntervals();
        for (Interval val : res)
            System.out.print(val + " ");

    }

    @Test
    public void testgR() {
        int[] nums1 = {1, 2, 1};
        int[] nums2 = {1, 3, 4, 2};
        arrayAlgoQuestion.nextGreaterElements(nums1);
    }

    @Test
    public void testMatrix() {
        int[][] matrix = {{1, 2, 3, 4}, {5, 6, 7, 8}, {9, 10, 11, 12}, {13, 14, 15, 16}};
        arrayAlgoQuestion.findDiagonalOrder(matrix);
    }

    @Test
    public void testPixel() {
        char[][] image = {{'0', '0', '1', '0'}, {'0', '1', '1', '0'}, {'0', '1', '0', '0'}, {'1', '1', '1', '1'}, {'1', '0', '0', '1'}};
        System.out.println(arrayAlgoQuestion.minArea(image, 3, 2));
    }

    @Test
    public void testWindow() {
        int[] nums = {1, 3, -1, -3, 5, 3, 6, 7};
        arrayAlgoQuestion.maxSlidingWindow(nums, 3);
    }

    @Test
    public void testAnother() {
        int[] nums = {3, 1};
        System.out.println(arrayAlgoQuestion.searchInSortedArrayAnotherVersion(nums, 3));
    }

    @Test
    public void testSortWithFrequ() {
        int[] nums = {2, 5, 2, 6, -1, 9999999, 5, 8, 8, 8};
        arrayAlgoQuestion.sortByFrequency(nums);
    }

    @Test
    public void testMergeSort() {
        int[] nums = {5, 2, 6, 1, 4};
        arrayAlgoQuestion.mergeSort(nums);
        System.out.println("---------");
        for (int x : nums)
            System.out.println(x);
    }

    @Test
    public void testMaximumSub() {
        int[] nums = {1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, -2, 1, -15, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1};
        ArrayList<Integer> res = new ArrayList<Integer>();
        for (int x : nums)
            res.add(x);
        arrayAlgoQuestion.maxTwoSubArrays(res);
    }

    @Test
    public void testContainsDuplicate() {
        int[] nums = {-3, 3};
        System.out.println(arrayAlgoQuestion.containsNearByAlmostDuplicate(nums, 2, 4));
    }

    @Test
    public void testReverse() {
        ContestQuestion cq = new ContestQuestion();
        int[] nums = {2147483647, 2147483647, -2147483647, -2147483647, -2147483647, 2147483647};
        cq.reversePairs(nums);
        cq.countSmaller(nums);
    }


}
