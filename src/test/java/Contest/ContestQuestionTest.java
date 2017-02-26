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
        String[] ans = {"ale", "apple", "monkey", "plea"};
        List<String> res = new ArrayList<>(Arrays.asList(ans));
        System.out.println(cq.findLongestWord("abpcplea", res));
    }


}
