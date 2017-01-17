package backtracking;

/**
 * Created by Tao on 1/16/2017.
 */

import org.junit.Before;
import org.junit.Test;

import java.util.*;

import static org.junit.Assert.*;

public class BacktrackTest {

    Backtrack bk = null;

    @Before
    public void setup() {
        bk = new Backtrack();
    }

    @Test
    public void testGenerate() {
        //bk.generateParenthesis(3);
        bk.generateParenthesisByStack(3);
    }

    @Test
    public void testTime() {
        bk.readBinaryWatch(1);
    }

    @Test
    public void testGrayCode() {
        List<Integer> res = bk.grayCode(3);
        for (int x : res) {
            System.out.println(x);
        }
    }

    @Test
    public void testPermute() {
        int nums[] = {1, 2, 3};
        bk.permute(nums);
    }

    @Test
    public void testPermuteUniq() {
        int nums[] = {1, 1, 2};
        bk.permuteUnique(nums);
    }

    @Test
    public void testlettercombination() {
        bk.letterCombinations("2");
    }

    @Test
    public void restoreIp() {
        bk.restoreIpAddresses("25525511135");
    }

    @Test
    public void testPalindrome() {
        bk.generatePalindromes("aabb");
    }

    @Test
    public void testFactor() {
        bk.getFactors(12);
    }

    @Test
    public void testTrie() {
        Trie t = new Trie();
        t.insert("so");
        assertEquals(t.search("soo"), false);
    }

    @Test
    public void testWordDictionary() {
        WordDictionary wd = new WordDictionary();
        wd.addWord("bad");
        wd.addWord("dad");
        wd.addWord("mad");
        assertEquals(wd.search("pad"), false);
        assertEquals(wd.search("bad"), true);
        assertEquals(wd.search(".ad"), true);
        assertEquals(wd.search("b.."), true);
    }

    @Test
    public void testFindwords() {
        String[] words = {"oath", "pea", "eat", "rain"};
        char[][] board = {{'o', 'a', 'a', 'n'}, {'e', 't', 'a', 'e'}, {'i', 'h', 'k', 'r'}, {'i', 'f', 'l', 'v'}};
        bk.findWords(board, words);
    }

    @Test
    public void testCanWin() {
        bk.canWin("++++");
    }

    @Test
    public void testWordBreak() {
        String[] ss = {"cat", "cats", "and", "sand", "dog"};
        List<String> res = new ArrayList<>(Arrays.asList(ss));
        bk.wordBreakII("catsanddog", res);
    }


}
