package backtracking;

/**
 * Created by Tao on 1/16/2017.
 */


import java.util.*;

//208 implement trie
class TrieNode {
    // Initialize your data structure here.
    boolean isEnd;
    TrieNode[] nodes = null;
    char content;

    public TrieNode() {
        isEnd = false;
        nodes = new TrieNode[26];
        content = ' ';
    }

    public TrieNode(char c) {
        isEnd = false;
        nodes = new TrieNode[26];
        content = c;
    }

    public TrieNode getNodes(char c) {
        return nodes != null ? nodes[c - 'a'] : null;
    }
}

class Trie {
    private TrieNode root;

    public TrieNode getRoot() {
        return root;
    }

    public Trie() {
        root = new TrieNode();
    }

    // Inserts a word into the trie.
    public void insert(String word) {
        int n = word.length();
        TrieNode cur = root;
        for (int i = 0; i < n; ++i) {
            if (cur.getNodes(word.charAt(i)) == null) {
                cur.nodes[word.charAt(i) - 'a'] = new TrieNode(word.charAt(i));
            }
            cur = cur.nodes[word.charAt(i) - 'a'];
        }
        cur.isEnd = true;
    }

    // Returns if the word is in the trie.
    public boolean search(String word) {
        int n = word.length();
        TrieNode cur = root;
        for (int i = 0; i < n; ++i) {
            if (cur.getNodes(word.charAt(i)) == null)
                return false;
            else
                cur = cur.nodes[word.charAt(i) - 'a'];
        }
        return cur.isEnd;
    }

    // Returns if there is any word in the trie
    // that starts with the given prefix.
    public boolean startsWith(String word) {
        int n = word.length();
        TrieNode cur = root;
        for (int i = 0; i < n; ++i) {
            if (cur.getNodes(word.charAt(i)) == null)
                return false;
            else
                cur = cur.nodes[word.charAt(i) - 'a'];
        }
        return true;
    }
}


class WordDictionary {


    private TrieNode root;

    public WordDictionary() {
        root = new TrieNode();
    }

    // Adds a word into the data structure.
    public void addWord(String word) {
        TrieNode cur = root;
        int n = word.length();
        for (int i = 0; i < n; ++i) {
            if (cur.getNodes(word.charAt(i)) == null) {
                cur.nodes[word.charAt(i) - 'a'] = new TrieNode(word.charAt(i));
            }
            cur = cur.getNodes(word.charAt(i));
        }
        cur.isEnd = true;
    }

    // Returns if the word is in the data structure. A word could
    // contain the dot character '.' to represent any one letter.
    public boolean search(String word) {
        return find(root, word, 0);
    }

    public boolean find(TrieNode node, String word, int pos) {
        if (pos == word.length())
            return node.isEnd;
        if (word.charAt(pos) == '.') {
            for (TrieNode subnode : node.nodes) {
                if (subnode != null && find(subnode, word, pos + 1))
                    return true;
            }
            return false;
        }
        return node.getNodes(word.charAt(pos)) == null ? false : find(node.getNodes(word.charAt(pos)), word, pos + 1);
    }

}


public class Backtrack {

    //22 generate parentheses
    public void generate(List<String> res, int left, int right, int n, String path) {
        if (left == right && left == n) {
            res.add(path);
            return;
        }
        System.out.println();
        if (left < n) {
            generate(res, left + 1, right, n, path + "(");
            System.out.println();
        }
        if (right < left) {
            generate(res, left, right + 1, n, path + ")");
            System.out.println();
        }

    }

    public List<String> generateParenthesis(int n) {
        List<String> res = new ArrayList<>();
        generate(res, 0, 0, n, "");
        return res;
    }

    //you can also use a stack to simulate

    public void backtracking(Stack<Integer> nums, Stack<Integer> stk, List<String> res, StringBuilder sb) {
        if (stk.isEmpty() && nums.isEmpty()) {
            res.add(sb.toString());
            return;
        }
        if (!nums.isEmpty()) {
            int top = nums.pop();
            stk.push(top);
            sb.append("(");
            backtracking(nums, stk, res, sb);
            nums.push(top);
            stk.pop();
            sb.deleteCharAt(sb.length() - 1);
        }
        if (!stk.isEmpty()) {
            int top = stk.pop();
            sb.append(")");
            backtracking(nums, stk, res, sb);
            stk.push(top);
            sb.deleteCharAt(sb.length() - 1);
        }
    }

    public List<String> generateParenthesisByStack(int n) {
        Stack<Integer> nums = new Stack<>();
        Stack<Integer> stk = new Stack<>();//simulate the skt
        for (int i = 1; i <= n; ++i)
            nums.push(i);
        List<String> res = new ArrayList<>();
        StringBuilder sb = new StringBuilder("");
        backtracking(nums, stk, res, sb);
        return res;
    }

    //401 binary watch
    public List<String> readBinaryWatch(int n) {
        List<String> time = new ArrayList<>();
        for (int h = 0; h < 12; ++h) {
            for (int m = 0; m <= 59; ++m) {
                if (Integer.bitCount(m) + Integer.bitCount(h) == n) {
                    time.add(String.format("%d:%02d", h, m));
                }
            }
        }
        return time;
    }

    //89 gray code
    public List<Integer> grayCode(int n) {
        List<Integer> res = new ArrayList<>();
        res.add(0);
        for (int i = 0; i < n; ++i) {
            int nn = res.size();
            for (int j = nn - 1; j >= 0; --j) {
                res.add(res.get(j) | (1 << i));
            }
        }
        return res;
    }

    //46 permutations

    public void permutedfs(List<List<Integer>> res, int[] nums, List<Integer> path) {
        if (path.size() == nums.length) {
            res.add(new ArrayList<>(path));
            return;
        }
        for (int i = 0; i < nums.length; ++i) {
            if (!path.contains(nums[i])) {
                path.add(nums[i]);
                permutedfs(res, nums, path);
                path.remove(path.size() - 1);
            }
        }

    }

    public List<List<Integer>> permute(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        List<Integer> path = new ArrayList<>();
        permutedfs(res, nums, path);
        return res;
    }

    //47 permutations with duplicate

    public void permuteUniqdfs(List<List<Integer>> res, int[] nums, List<Integer> path) {
        if (path.size() == nums.length) {
            res.add(new ArrayList<>(path));
            return;
        }
        for (int i = 0; i < nums.length; ++i) {
            if (i == 0 || nums[i] != nums[i - 1]) {
                int c1 = 0, c2 = 0;
                for (int x : nums)
                    if (x == nums[i])
                        c1++;
                for (int x : path) {
                    if (x == nums[i])
                        c2++;
                }
                if (c1 > c2) {
                    path.add(nums[i]);
                    permuteUniqdfs(res, nums, path);
                    path.remove(path.size() - 1);
                }
            }
        }
    }

    public List<List<Integer>> permuteUnique(int[] nums) {
        List<List<Integer>> res = new ArrayList<>();
        List<Integer> path = new ArrayList<>();
        Arrays.sort(nums);
        permuteUniqdfs(res, nums, path);
        return res;
    }

    //17 letter of combination of a phone number
    public void backtrackLetter(List<String> res, String[] strs, String digits, int pos, String path) {
        if (path.length() == digits.length()) {
            res.add(path);
            return;
        }
        if (pos >= digits.length())
            return;
        for (int i = 0; i < strs[digits.charAt(pos) - '2'].length(); ++i) {
            backtrackLetter(res, strs, digits, pos + 1, path + strs[digits.charAt(pos) - '2'].charAt(i));
        }
    }

    public List<String> letterCombinations(String digits) {
        String[] strs = {"abc", "def", "ghi", "jkl", "mno", "pqrs", "tuv", "wxyz"};
        List<String> res = new ArrayList<>();
        if (digits.length() == 0)
            return res;
        backtrackLetter(res, strs, digits, 0, "");
        return res;
    }

    //52 NqueensII

    public boolean check(int[] path, int k) {
        for (int i = 0; i < k; ++i) {
            if (path[k] == path[i] || k - i == Math.abs(path[i] - path[k]))
                return false;
        }
        return true;
    }

    public void backtrackQueens(int[] path, int n, int[] res, int pos) {
        if (pos == n) {
            res[0]++;
            return;
        }
        for (int i = 0; i < n; ++i) {
            path[pos] = i;
            if (check(path, pos))
                backtrackQueens(path, n, res, pos + 1);
        }
    }

    public int totalNQueensII(int n) {
        int[] res = {0};
        int[] path = new int[n];
        backtrackQueens(path, n, res, 0);
        return res[0];
    }

    //51 Nqueens I

    public void backtrackQueensII(List<List<String>> res, int[] path, int n, int pos) {
        if (pos == n) {
            String tmp = "";
            List<String> tmps = new ArrayList<>();
            for (int i = 0; i < n; ++i) {
                tmp = "";
                for (int j = 0; j < n; ++j) {
                    tmp += path[i] == j ? "Q" : ".";
                }
                tmps.add(tmp);
            }
            res.add(tmps);
            return;
        }
        for (int i = 0; i < n; ++i) {
            path[pos] = i;
            if (check(path, pos))
                backtrackQueensII(res, path, n, pos + 1);
        }
    }

    public List<List<String>> solveNQueensI(int n) {
        List<List<String>> res = new ArrayList<>();
        int[] path = new int[n];
        backtrackQueensII(res, path, n, 0);
        return res;
    }


    //93 restore Ip address

    public boolean validIp(String str) {
        int n = str.length();
        if (n > 1 && str.charAt(0) == '0') return false;
        return str.length() <= 3 && Integer.valueOf(str) <= 255;
    }


    public void restoreIpDfs(List<String> res, String s, List<String> path, int pos) {
        if (s.length() == pos && path.size() == 4) {
            String ss = "";
            for (String str : path) {
                ss += str + ".";
            }
            res.add(String.join(".", path));
            res.add(ss.substring(0, ss.length() - 1));
            return;
        }
        if (path.size() > 4)
            return;
        for (int i = pos + 1; i <= s.length(); ++i) {
            String substr = s.substring(pos, i);
            if (validIp(substr)) {
                path.add(substr);
                restoreIpDfs(res, s, path, i);
                path.remove(path.size() - 1);
            }
        }
    }

    public List<String> restoreIpAddresses(String s) {
        List<String> res = new ArrayList<>();
        List<String> path = new ArrayList<>();
        if (s.length() > 12 || s.length() < 4)//very important if the length is very long, how should you do;
            return res;
        restoreIpDfs(res, s, path, 0);
        return res;
    }

    ////266 palindrome permutation
    public boolean canPermutePalindrome(String s) {
        int[] count = new int[256];
        int n = s.length();
        for (int i = 0; i < n; ++i) {
            count[s.charAt(i)]++;
        }
        int numberOfOddNumber = 0;
        for (int i = 0; i < 256; ++i) {
            if ((count[i] & 0x1) == 1) {
                numberOfOddNumber++;
                if (numberOfOddNumber > 1)
                    return false;
            }
        }
        return true;
    }

    //267 Palindrome Permutation II


    public void PalindromeDfs(List<String> res, String sb, String path, int num, char special) {
        if (path.length() == sb.length()) {
            String leftPart = new StringBuilder(path).reverse().toString();
            if (num != 0) {
                res.add(path + special + leftPart);
            } else
                res.add(path + leftPart);
            return;
        }

        for (int i = 0; i < sb.length(); ++i) {
            if (i > 0 && sb.charAt(i) == sb.charAt(i - 1))
                continue;
            int c1 = 0, c2 = 0;
            for (int j = 0; j < sb.length(); ++j)
                if (sb.charAt(i) == sb.charAt(j))
                    c1++;
            for (int j = 0; j < path.length(); ++j)
                if (sb.charAt(i) == path.charAt(j))
                    c2++;
            if (c1 > c2) {
                PalindromeDfs(res, sb, path + sb.charAt(i), num, special);
            }
        }
    }

    public List<String> generatePalindromes(String s) {
        List<String> res = new ArrayList<>();
        int[] count = new int[256];
        int n = s.length();
        for (int i = 0; i < n; ++i) {
            count[s.charAt(i)]++;
        }
        int numberOfOddNumber = 0;
        StringBuilder sb = new StringBuilder("");
        char special = ' ';
        for (int i = 0; i < 256; ++i) {
            if (count[i] > 0) {
                for (int j = 1; j <= count[i] / 2; ++j)
                    sb.append((char) i);
                if ((count[i] & 0x1) == 1) {
                    special = (char) i;
                    numberOfOddNumber++;
                    if (numberOfOddNumber > 1)
                        return res;
                }
            }
        }

        PalindromeDfs(res, sb.toString(), "", numberOfOddNumber, special);
        return res;
    }

    //254 factor combinations

    public void getFactorsBacktrack(List<List<Integer>> res, List<Integer> path, List<Integer> candidates, int n, int pos) {
        if (n == 1) {
            res.add(new ArrayList<>(path));
            return;
        }
        for (int i = pos; i < candidates.size(); ++i) {
            if (n >= candidates.get(i)) {
                if (n % candidates.get(i) == 0) {//this is very important
                    path.add(candidates.get(i));
                    getFactorsBacktrack(res, path, candidates, n / candidates.get(i), i);
                    path.remove(path.size() - 1);
                }
            } else
                return;
        }
    }

    public List<List<Integer>> getFactors(int n) {
        List<Integer> candidates = new ArrayList<>();
        for (int i = 2; i <= n / i; ++i) {
            if (n % i == 0) {
                candidates.add(i);
                if (i != n / i)
                    candidates.add(n / i);
            }
        }
        List<List<Integer>> res = new ArrayList<>();
        if (candidates.size() == 0)
            return res;
        List<Integer> path = new ArrayList<>();
        Collections.sort(candidates);

        getFactorsBacktrack(res, path, candidates, n, 0);
        return res;
    }


    //79 word search I
    public boolean dfs(char[][] board, String word, int pos, int x, int y) {
        //judge this first, otherwise corn case ['a'] and 'a'
        //
        if (pos == word.length()) {
            return true;
        }
        if (x < 0 || y < 0 || x >= board.length || y >= board[0].length || board[x][y] == '#')//already used
            return false;
        if (board[x][y] != word.charAt(pos))
            return false;
        char c = word.charAt(pos);
        board[x][y] = '#';
        boolean exists = dfs(board, word, pos + 1, x + 1, y) || dfs(board, word, pos + 1, x - 1, y) || dfs(board, word, pos + 1, x, y + 1) || dfs(board, word, pos + 1, x, y - 1);
        board[x][y] = c;
        return exists;
    }

    public boolean exist(char[][] board, String word) {
        if (board.length == 0 || board[0].length == 0) {
            return false;
        }
        int m = board.length, n = board[0].length;
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (dfs(board, word, 0, i, j))
                    return true;
            }
        }
        return false;
    }
    //212 word search II


    public void findWordsBacktrack(char[][] board, TrieNode node, int x, int y, String path, List<String> res) {
        if (node != null && node.isEnd) {
            res.add(path);
            node.isEnd = false;
        }
        if (node == null || x < 0 || x >= board.length || y < 0 || y >= board[0].length || board[x][y] == '@')
            return;
        char c = board[x][y];
        board[x][y] = '@';
        findWordsBacktrack(board, node.getNodes(c), x + 1, y, path + c, res);
        findWordsBacktrack(board, node.getNodes(c), x - 1, y, path + c, res);
        findWordsBacktrack(board, node.getNodes(c), x, y + 1, path + c, res);
        findWordsBacktrack(board, node.getNodes(c), x, y - 1, path + c, res);
        board[x][y] = c;
    }

    public List<String> findWords(char[][] board, String[] words) {
        Trie t = new Trie();
        for (String word : words) {
            t.insert(word);
        }
        List<String> res = new ArrayList<>();
        if (board.length == 0 || board[0].length == 0)
            return res;
        int m = board.length, n = board[0].length;
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                findWordsBacktrack(board, t.getRoot(), i, j, "", res);
            }
        }
        return res;
    }

    //464 can I win TLE way

    public boolean canIWinHelper(List<Integer> res, int desiredTotal) {
        if (!res.isEmpty() && res.get(res.size() - 1) >= desiredTotal)
            return true;
        for (int i = 0; i < res.size(); ++i) {
            int removed = res.remove(i);
            boolean win = !canIWinHelper(res, desiredTotal - removed);
            res.add(i, removed);
            if (win) {
                return true;
            }
        }
        return false;
    }

    public boolean canIWin(int maxChoosableInteger, int desiredTotal) {
        List<Integer> res = new ArrayList<>();
        for (int i = 1; i <= maxChoosableInteger; ++i) {
            res.add(i);
        }
        return canIWinHelper(res, desiredTotal);
    }

    //actually you can use hashmap to store the state
    public boolean canIWinWithHashmap(int maxChooseableInteger, int desiredTotal) {
        int value = (1 + maxChooseableInteger) * (maxChooseableInteger) / 2;
        if (value < desiredTotal)
            return false;
        boolean[] used = new boolean[maxChooseableInteger + 1];
        Map<Integer, Boolean> map = new HashMap<>();
        return canIWinWithHashmapHelper(used, desiredTotal, map);
    }

    public boolean canIWinWithHashmapHelper(boolean[] used, int desiredTotal, Map<Integer, Boolean> map) {
        if (desiredTotal <= 0)
            return false;
        int val = formatToInteger(used);
        if (map.containsKey(val))
            return map.get(val);
        for (int i = 1; i < used.length; ++i) {
            if (!used[i]) {
                used[i] = true;
                if (!canIWinWithHashmapHelper(used, desiredTotal - i, map)) {
                    map.put(val, true);
                    used[i] = false;
                    return true;
                }
                used[i] = false;
            }
        }
        map.put(val, false);
        return false;
    }

    public int formatToInteger(boolean[] used) {
        int res = 0;
        for (boolean b : used) {
            res <<= 1;
            if (b)
                res |= 1;
        }
        return res;
    }






    //294 flip gameII
    public boolean canWin(String s) {
        if (s == null || s.length() < 2)
            return false;
        for (int i = 0; i < s.length() - 1; ++i) {
            if (s.startsWith("++", i)) {
                StringBuilder sb = new StringBuilder(s);//substring is lower, so you can use stringbuilder
                sb.setCharAt(i, '-');
                sb.setCharAt(i + 1, '-');
                if (!canWin(sb.toString()))
                    return true;
            }
        }
        return false;
    }

    public boolean MycanWin(String s) {
        if (s == null || s.length() < 2)
            return false;
        for (int i = 0; i < s.length() - 1; ++i) {
            if (s.substring(i, i + 2).equals("++")) {
                StringBuilder sb = new StringBuilder(s);
                sb.setCharAt(i, '-');
                sb.setCharAt(i + 1, '-');
                if (!MycanWin(sb.toString()))
                    return true;
            }
        }
        return false;
    }


    //the fastest way to solve this problem
    public boolean helper(String s, Map<String, Boolean> map) {
        if (map.containsKey(s))
            return map.get(s);
        for (int i = 0; i < s.length() - 1; ++i) {
            if (s.startsWith("++", i)) {
                StringBuilder sb = new StringBuilder(s);
                sb.setCharAt(i, '-');
                sb.setCharAt(i + 1, '-');
                if (!helper(sb.toString(), map)) {
                    map.put(s, true);
                    return true;
                }
            }
        }
        map.put(s, false);
        return false;
    }

    public boolean MycanWinWithHashMap(String s) {
        if (s == null || s.length() < 2)
            return false;
        Map<String, Boolean> map = new HashMap<>();
        return helper(s, map);
    }



    //139 word break I
    public boolean wordBreak(String s, List<String> wordDict) {
        int n = s.length();
        boolean[] dp = new boolean[n + 1];
        dp[0] = true;
        Set<String> sets = new HashSet<>(wordDict);
        for (int i = 1; i <= n; ++i) {
            for (int j = 0; j < i; ++j) {
                if (dp[j] && sets.contains(s.substring(j, i)))
                    dp[i] = true;
            }
        }
        return dp[n];
    }

    //140 wordbreak II
    //TLE
    public List<String> wordbreakIIdfs(String s, Set<String> sets, Map<String, LinkedList<String>> map, int pos) {
        if (map.containsKey(s.substring(pos)))//easy to overflow, so you'd better prepare not use string to find. There are just few words in set. So search in a set is much faster.
            return map.get(s.substring(pos));
        LinkedList<String> res = new LinkedList<>();
        if (s.length() == pos) {
            res.add("");
            return res;
        }
        for (int i = pos + 1; i <= s.length(); ++i) {
            String substr = s.substring(pos, i);
            if (sets.contains(substr)) {
                List<String> sublist = wordbreakIIdfs(s, sets, map, i);
                for (String sub : sublist) {
                    res.add(substr + (sub.isEmpty() ? "" : " ") + sub);
                }
            }
        }
        map.put(s, res);
        return res;
    }

    public List<String> wordBreakII(String s, List<String> wordDict) {
        Set<String> sets = new HashSet<>(wordDict);
        return wordbreakIIdfs(s, sets, new HashMap<String, LinkedList<String>>(), 0);
    }

    //version 2
    List<String> DFS(String s, Set<String> wordDict, HashMap<String, LinkedList<String>> map) {
        if (map.containsKey(s))
            return map.get(s);

        LinkedList<String> res = new LinkedList<String>();
        if (s.length() == 0) {
            res.add("");// actually "" corresponding to {""}, moreover, if you just return {}, then there is no different with can not find anything. So you must return something
            return res;
        }
        for (String word : wordDict) {
            if (s.startsWith(word)) {
                List<String> sublist = DFS(s.substring(word.length()), wordDict, map);
                for (String sub : sublist)
                    res.add(word + (sub.isEmpty() ? "" : " ") + sub);
            }
        }
        map.put(s, res);
        return res;
    }

    public List<String> wordBreakII2(String s, List<String> wordDict) {
        Set<String> sets = new HashSet<>(wordDict);
        return DFS(s, sets, new HashMap<String, LinkedList<String>>());
    }


    public boolean isPalindrome(String s) {
        int n = s.length(), i = 0;
        while (i < n / 2) {
            if (s.charAt(i) != s.charAt(n - i - 1))
                return false;
            i++;
        }
        return true;
    }


    public void getPalindromeDfs(String s, List<List<String>> res, int pos, List<String> path) {
        if (pos == s.length()) {
            res.add(new ArrayList<>(path));
            return;
        }
        for (int i = pos + 1; i <= s.length(); ++i) {
            String substr = s.substring(pos, i);
            if (isPalindrome(substr)) {
                path.add(substr);
                getPalindromeDfs(s, res, i, path);
                path.remove(path.size() - 1);
            }
        }
    }

    //131 Palindrome Partitioning
    public List<List<String>> partition(String s) {
        int n = s.length();
        List<List<String>> res = new ArrayList<>();
        getPalindromeDfs(s, res, 0, new ArrayList<String>());
        return res;
    }

    //you can also use dp to deal with first
    public void partitionWithDpdfs(List<List<String>> res, String s, boolean[][] dp, int pos, ArrayList<String> path) {
        if (pos == s.length()) {
            res.add(new ArrayList<>(path));
            return;
        }
        for (int i = pos; i < s.length(); ++i) {
            if (dp[pos][i]) {
                path.add(s.substring(pos, i + 1));
                partitionWithDpdfs(res, s, dp, i + 1, path);
                path.remove(path.size() - 1);
            }
        }
    }

    public List<List<String>> partitionWithDp(String s) {
        List<List<String>> res = new ArrayList<>();
        int n = s.length();
        boolean[][] dp = new boolean[n][n];
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j <= i; ++j) {
                if (s.charAt(i) == s.charAt(j) && ((i - j <= 2) || dp[j + 1][i - 1]))
                    dp[j][i] = true;
            }
        }
        partitionWithDpdfs(res, s, dp, 0, new ArrayList<String>());
        return res;
    }

    //follow up palindrome partitioning II
    public int minCut(String s) {
        int n = s.length();
        boolean[][] dp = new boolean[n][n];
        int[] res = new int[n + 1];
        Arrays.fill(res, -1);
        for (int i = 0; i < n; ++i) {
            res[i + 1] = res[i] + 1;
            dp[i][i] = true;
            for (int j = 0; j < i; ++j) {
                if (s.charAt(i) == s.charAt(j) && ((i - j <= 2) || dp[j + 1][i - 1])) {
                    dp[j][i] = true;
                    if (j == 0)
                        res[i + 1] = 0;
                    else
                        res[i + 1] = Math.min(res[i + 1], res[j] + 1);
                    //break;
                }
            }
        }
        return res[n];

    }

    public boolean helper(List<Integer> list, int[] res, int pos) {
        if (list.isEmpty()) {
            int index = pos == 0 ? 1 : 0;
            return res[pos] >= res[index];
        }
        int val = list.remove(0);
        res[pos] += val;
        boolean win = !helper(list, res, pos == 0 ? 1 : 0);
        list.add(0, val);
        res[pos] -= val;
        if (win)
            return true;
        val = list.remove(list.size() - 1);
        res[pos] += val;
        win = !helper(list, res, pos == 0 ? 1 : 0);
        list.add(val);
        res[pos] -= val;
        return win;

    }

    public boolean PredictTheWinner(int[] nums) {
        int[] res = new int[2];
        List<Integer> li = new ArrayList<>();
        for (int x : nums)
            li.add(x);
        if (nums.length <= 2)
            return true;
        return helper(li, res, 0);
    }

}
