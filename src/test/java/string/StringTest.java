package string;

/**
 * Created by Tao on 1/13/2017.
 */

import com.sun.tools.doclets.formats.html.SourceToHTMLConverter;
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

    @Test
    public void testNumberofSegement() {
        System.out.println(sq.countSegments("Hello, my name is John     hah     haha       haa"));
    }

    @Test
    public void testshiftGroup() {
        String[] strs = {"abc", "bcd", "acef", "xyz", "az", "ba", "a", "z"};
        sq.groupStrings(strs);


    }

    @Test
    public void testValidWord() {
        List<String> str = new ArrayList<>();
        str.add("balle");
        str.add("aset");
        str.add("le");
        str.add("lt");
        str.add("lt");
        sq.validWordSquare(str);
    }

    @Test
    public void testMagic() {
        System.out.println(sq.magicalString(16));
    }



}
