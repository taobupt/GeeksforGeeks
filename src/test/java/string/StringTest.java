package string;

/**
 * Created by Tao on 1/13/2017.
 */

import org.junit.Before;
import org.junit.Test;

import java.util.*;

import static org.junit.Assert.*;

public class StringTest {
    StringQuestion sq = null;

    @Before
    public void setup() {
        sq = new StringQuestion();
    }

    @Test
    public void testLengthofLastWord() {
        assertEquals(sq.lengthOfLastWord("abcdef fd   ee    ea"), 2);
    }

    @Test
    public void testValidParenthese() {
        assertEquals(sq.isValid(""), true);
    }

    @Test
    public void testValidPalindrome() {
        assertEquals(sq.isPalindrome("race a car"), false);
    }

    @Test
    public void testReverseString() {
        System.out.println(sq.reverseStringSolution2("hello"));
    }


}
