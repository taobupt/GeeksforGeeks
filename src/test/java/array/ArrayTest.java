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

}
