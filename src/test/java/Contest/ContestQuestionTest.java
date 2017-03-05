package Contest;

/**
 * Created by tao on 2/16/17.
 */

import leetcodeContest.ContestQuestion;
import org.junit.Before;
import org.junit.Test;

import java.beans.Transient;
import java.util.*;

import static org.junit.Assert.*;

public class ContestQuestionTest {


    public ContestQuestion cq = null;

    @Before
    public void setup() {
        cq = new ContestQuestion();
    }

    @Test
    public void testMegresort() {
        int[] nums = {4, 6, 8, 1, 2, 3};
        cq.mergeSort(nums);
    }

    @Test
    public void testBeautiful() {
        System.out.println(cq.countArrangement(15));
    }

    @Test
    public void testFindRotate() {
        System.out.println(cq.findRotateStepsDP("godding", "gd"));

    }

}
