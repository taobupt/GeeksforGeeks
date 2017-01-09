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
}
