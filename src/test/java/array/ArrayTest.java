package array;

/**
 * Created by tao on 1/9/17.
 */

import org.junit.Before;
import org.junit.Test;

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
        System.out.println(arrayAlgoQuestion.moveCharacter(s));
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
}
