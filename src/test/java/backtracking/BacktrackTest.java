package backtracking;

/**
 * Created by Tao on 1/16/2017.
 */

import org.junit.Before;
import org.junit.Test;

import java.io.BufferedReader;
import java.io.FileReader;
import java.util.*;

import static org.junit.Assert.*;

public class BacktrackTest {

    Backtrack bk = null;

    public List<String> readTestCase(String fileName) {
        List<String> res = new ArrayList<>();
        try {

            BufferedReader reader = new BufferedReader(new FileReader(fileName));
            String str;
            while ((str = reader.readLine()) != null) {
                res.add(str);
            }
            reader.close();
        } catch (Exception e) {
            e.printStackTrace();
        }
        return res;
    }


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


        List<String> testCase = readTestCase("K:\\github\\GeeksforGeeks\\duipai\\testcase.txt");
        for (String str : testCase) {
            //System.out.println("mycanwin"+bk.MycanWinWithHashMap(str));
            assertEquals(bk.MycanWinWithHashMap(str), bk.canWin(str));
        }
        //System.out.println(bk.MycanWinWithHashMap("++----+---+--+++-+"));
        //System.out.println(bk.canWin("++----+---+--+++-+"));
    }

    @Test
    public void testWordBreak() {
        String[] ss = {"cat", "cats", "and", "sand", "dog"};
        List<String> res = new ArrayList<>(Arrays.asList(ss));
        bk.wordBreakII("catsanddog", res);
    }

    @Test
    public void testPartitionPalindrome() {
        bk.partition("aab");
    }

    @Test
    public void testPartitionPalindromeCut() {
        System.out.println(bk.minCut("abcdefgfedc"));
    }

    @Test
    public void testJudge() {
        int[] nums = {1, 1, 1};
        System.out.println(bk.PredictTheWinner(nums));
    }


}
