package string;

import java.util.*;

/**
 * Created by Tao on 1/13/2017.
 */

class Codec {

    // Encodes a list of strings to a single string.
    public String encode(List<String> strs) {
        StringBuilder sb = new StringBuilder();
        for (String str : strs)
            sb.append(str.length() + "@" + str);
        return sb.toString();
    }


    // Decodes a single string to a list of strings.
    public List<String> decode(String s) {
        List<String> res = new ArrayList<>();
        int n = s.length();
        int j = 0;
        while (j < n) {
            int index = s.indexOf('@', j);
            int val = Integer.valueOf(s.substring(j, index));
            res.add(s.substring(index + 1, index + 1 + val));
            j = index + 1 + val;
        }
        return res;
    }
}


public class StringQuestion {

    //58 length of last word
    public int lengthOfLastWord(String s) {
        int res = 0, last = s.length() - 1;
        while (last >= 0 && Character.isSpaceChar(s.charAt(last)))
            last--;
        while (last >= 0 && !Character.isSpaceChar(s.charAt(last))) {
            last--;
            res++;
        }
        return res;
    }

    //second solution
    public int lengthOfLastWordSolution2(String s) {
        return s.trim().length() - s.trim().lastIndexOf(" ") - 1;
    }

    //lt 20
    //for(char c:String) is not allowed, you should convert it to charArray first String.toCharArray()
    public boolean isValid(String s) {
        Stack<Character> stk = new Stack<>();
        int n = s.length();
        for (int i = 0; i < n; ++i) {
            switch (s.charAt(i)) {
                case '(':
                case '[':
                case '{':
                    stk.push(s.charAt(i));
                    break;
                case '}':
                    if (stk.isEmpty() || stk.peek() != '{')
                        return false;
                    stk.pop();
                    break;
                case ')':
                    if (stk.isEmpty() || stk.peek() != '(')
                        return false;
                    stk.pop();
                    break;
                case ']':
                    if (stk.isEmpty() || stk.peek() != '[')
                        return false;
                    stk.pop();
                    break;
            }
        }
        return stk.isEmpty();
    }

    //some interesting solution
    public boolean isValidConcise(String s) {
        Stack<Character> stk = new Stack<>();
        for (char c : s.toCharArray()) {
            if (c == '(')
                stk.push(')');
            else if (c == '[')
                stk.push(']');
            else if (c == '{')
                stk.push('}');
            else if (stk.isEmpty() || stk.pop() != c)
                return false;
        }
        return stk.isEmpty();
    }

    //lt 125 valid palindrome
    public boolean isPalindrome(String s) {
        int begin = 0, end = s.length() - 1;
        while (begin < end) {
            while (begin < end && !Character.isLetterOrDigit(s.charAt(begin)))
                begin++;
            while (begin < end && !Character.isLetterOrDigit(s.charAt(end)))
                end--;
            if (Character.toLowerCase(s.charAt(begin++)) != Character.toLowerCase(s.charAt(end--)))
                return false;
        }
        return true;
    }

    //344 reverse string
    //String buffer is slow, but charArray is faster
    public String reverseString(String s) {
        StringBuffer sb = new StringBuffer(s);
        int n = sb.length();
        for (int i = 0; i < n / 2; ++i) {
            char c = sb.charAt(i);
            sb.setCharAt(i, sb.charAt(n - i - 1));
            sb.setCharAt(n - i - 1, c);
        }
        return sb.toString();
    }

    //you can choose another way, convert to chararray
    public String reverseStringSolution2(String s) {
        char[] word = s.toCharArray();
        int n = s.length();
        for (int i = 0; i < n / 2; ++i) {
            char c = word[i];
            word[i] = word[n - i - 1];
            word[n - i - 1] = c;
        }
        return new String(word);
    }

    public String reverseStringSolution3(String s) {
        int n = s.length();
        if (n <= 1)
            return s;
        String leftstr = s.substring(0, n / 2);
        String rightstr = s.substring(n / 2, n);
        return reverseStringSolution3(rightstr) + reverseStringSolution3(leftstr);
        //The recurrence equation is `T(n) = 2 * T(n/2) + O(n)`.is due to the fact that concatenation function takes linear time. The recurrence equation can be solved to get `O(n * log(n))`.
    }

    //lt 383 ransom note
    public boolean canConstruct(String ransomNote, String magazine) {
        //if the String is much more than 256, you should use hashmap,otherwise, char[] is enough
        int[] count = new int[256];
        int m = ransomNote.length(), n = magazine.length();
        for (int i = 0; i < n; ++i)
            count[magazine.charAt(i)]++;
        for (int i = 0; i < m; ++i) {
            count[ransomNote.charAt(i)]--;
            if (count[ransomNote.charAt(i)] < 0)
                return false;
        }
        return true;
    }

    //13 roman to integer
    public int romanToInt(String s) {
        Map<Character, Integer> mp = new HashMap<>();
        mp.put('I', 1);
        mp.put('V', 5);
        mp.put('X', 10);
        mp.put('L', 50);
        mp.put('C', 100);
        mp.put('D', 500);
        mp.put('M', 1000);
        int n = s.length();
        if (n == 0)
            return 0;
        int res = mp.get(s.charAt(n - 1));
        for (int i = n - 2; i >= 0; --i) {
            res += mp.get(s.charAt(i)) >= mp.get(s.charAt(i + 1)) ? mp.get(s.charAt(i)) : -mp.get(s.charAt(i));
        }
        return res;
    }

    //another interesting solution
    public int romanToIntSolution2(String s) {
        int res = 0;
        int n = s.length();
        for (int i = n - 1; i >= 0; --i) {
            char c = s.charAt(i);
            switch (c) {
                case 'I':
                    res += (res >= 5 ? -1 : 1);
                    break;
                case 'V':
                    res += 5;
                    break;
                case 'X':
                    res += 10 * (res >= 50 ? -1 : 1);
                    break;
                case 'L':
                    res += 50;
                    break;
                case 'C':
                    res += 100 * (res >= 500 ? -1 : 1);
                    break;
                case 'D':
                    res += 500;
                    break;
                case 'M':
                    res += 1000;
                    break;
            }
        }
        return res;
    }

    //67 add binary
    public String addBinary(String a, String b) {
        int m = a.length(), n = b.length(), carry = 0;
        int i = m - 1, j = n - 1;
        StringBuffer sb = new StringBuffer("");
        while (i >= 0 || j >= 0 || carry != 0) {
            carry += (i >= 0 ? a.charAt(i--) - '0' : 0) + (j >= 0 ? b.charAt(j--) - '0' : 0);
            sb.append((char) (carry % 2 + '0'));
            carry /= 2;
        }
        return sb.reverse().toString();
        //if you don't want to use sb.reverse you can use sb.insert(0,char)// but you have to shift other character aways
        //Character.getNumericValue()
    }

    //293 flip game
    public List<String> generatePossibleNextMove(String s) {
        StringBuffer sb = new StringBuffer(s);
        List<String> res = new ArrayList<>();
        int n = s.length();
        for (int i = 1; i < n; ++i) {
            if (sb.substring(i - 1, i + 1).equals("++")) {
                sb.setCharAt(i, '-');
                sb.setCharAt(i - 1, '-');
                res.add(sb.toString());
                sb.setCharAt(i, '+');
                sb.setCharAt(i - 1, '+');
            }
        }
        return res;
    }

    //345 reverse vowels of a string
    public boolean isVowels(char c) {
        char cc = Character.toLowerCase(c);
        return cc == 'a' || cc == 'e' || cc == 'i' || cc == 'o' || cc == 'u';
    }

    public String reverseVowels(String s) {
        StringBuffer sb = new StringBuffer(s);
        int begin = 0, end = s.length() - 1;
        while (begin < end) {
            while (begin < end && !isVowels(sb.charAt(end)))
                end--;
            while (begin < end && !isVowels(sb.charAt(begin)))
                begin++;
            //here you can add begin<end or nothing, you think a little
            char c = sb.charAt(begin);
            sb.setCharAt(begin++, sb.charAt(end));
            sb.setCharAt(end--, c);
        }
        return sb.toString();
    }

    //28 implement strStr()
    public int strStr(String haystack, String needle) {
        int n = haystack.length(), m = needle.length();
        for (int i = 0; i <= n - m; ++i) {//do not forget the n-m not n
            if (haystack.substring(i, i + m).equals(needle)) {
                return i;
            }
        }
        return -1;
    }

    //or you can compare character by character
    public int strStrCharacterByCharacter(String haystack, String needle) {
        int n = haystack.length(), m = needle.length();
        for (int i = 0; i <= n - m; ++i) {//do not forget the n-m not n
            int j = i;
            for (; j < i + m; ++j) {
                if (haystack.charAt(j) != needle.charAt(j - i))
                    break;
            }
            if (j == i + m)
                return i;
        }
        return -1;
    }

    //14 longest common prefix
    public String longestCommonPrefix(String[] strs) {
        int n = strs.length;
        if (n == 0 || strs[0].length() == 0)
            return "";
        String res = "";
        int m = strs[0].length();
        for (int i = 1; i <= m; ++i) {
            res = strs[0].substring(0, i);
            for (int j = 1; j < n; ++j) {
                if (strs[j].length() >= res.length() && strs[j].substring(0, res.length()).equals(res))
                    continue;
                else
                    return res.substring(0, res.length() - 1);
            }
        }
        return res;
    }

    //you can use some funciton of java
    public String longestCommonPrefixConcise(String[] strs) {
        if (strs.length == 0) return "";
        String pre = strs[0];
        for (int i = 1; i < strs.length; i++)
            while (strs[i].indexOf(pre) != 0)
                pre = pre.substring(0, pre.length() - 1);
        return pre;
    }

    //434 number of segments in a string
    public int countSegments(String s) {
        String[] res = s.split(" ");
        int cnt = 0;
        for (String str : res) {
            if (!str.equals(""))
                cnt++;
        }
        return cnt;
    }

    //another version
    //2 ms version
    public int countSegmentsNotUseRegex(String s) {
        int n = s.length();
        int begin = 0, cnt = 0;
        while (begin < n) {
            while (begin < n && Character.isSpaceChar(s.charAt(begin)))
                begin++;
            if (begin < n)
                cnt++;
            while (begin < n && !Character.isSpaceChar(s.charAt(begin)))
                begin++;
        }
        return cnt;
    }

    //459 repeated substring pattern
    //you can sorted by brute force
    //pick every substring until n/2 and then check for all possibility
    //The length of the repeating substring must be a divisor of the length of the input string
    public boolean repeatedSubstringPattern(String str) {
        int n = str.length();
        for (int i = n / 2; i >= 1; --i) {
            if (n % i == 0) {
                int m = n / i;
                String sub = str.substring(0, i);
                StringBuilder sb = new StringBuilder();
                for (int j = 0; j < m; ++j) {
                    sb.append(sub);
                }
                if (sb.toString().equals(str))
                    return true;
            }
        }
        return false;
    }

    //249 group shifted strings
    public String changeStandard(String s) {
        StringBuilder sb = new StringBuilder(s);
        int len = sb.charAt(0) - 'a', n = s.length();
        for (int i = 0; i < n; ++i) {
            sb.setCharAt(i, (char) ((sb.charAt(i) - len + 26 - 'a') % 26 + 'a'));
        }
        return sb.toString();
    }

    public List<List<String>> groupStrings(String[] strings) {
        Map<String, List<String>> mp = new HashMap<>();
        int n = strings.length;
        for (int i = 0; i < n; ++i) {
            String sb = changeStandard(strings[i]);
            if (mp.containsKey(sb)) {
                mp.get(sb).add(strings[i]);
            } else {
                List<String> tmp = new ArrayList<>();
                tmp.add(strings[i]);
                mp.put(sb, tmp);
            }
        }
        List<List<String>> res = new ArrayList<>();
        for (Map.Entry<String, List<String>> entry : mp.entrySet()) {
            res.add(entry.getValue());
        }
        return res;
    }

    //157 read N characters given read4
    private int read4(char[] buf) {
        return 0;
    }

    public int read(char[] buf, int n) {
        char[] buffer = new char[4];
        int res = 0;
        while (n > res) {
            int num = read4(buffer);
            int length = Math.min(num, n - res);
            for (int i = 0; i < length; ++i) {
                buf[res + i] = buffer[i];
            }
            res += length;
            if (num < 4)
                break;
        }
        return res;
    }

    //158 read many times//    interesting
    private int buffCnt = 0;
    private int buffPtr = 0;
    private char[] buff = new char[4];

    public int readII(char[] buf, int n) {
        int cnt = 0;
        boolean hasNext = true;
        while (cnt < n && hasNext) {
            if (buffPtr == 0)
                buffCnt = read4(buff);
            if (buffCnt < 4)
                hasNext = false;
            while (cnt < n && buffPtr < buffCnt) {
                buf[cnt++] = buff[buffPtr++];
            }
            if (buffPtr == buffCnt)
                buffPtr = 0;
        }
        return cnt;
    }

    //76 Minimum Window Substring
    public String minWindow(String s, String t) {
        int m = s.length(), cnt = t.length(), n = t.length(), begin = 0, end = 0, start = 0, length = Integer.MAX_VALUE;
        char[] mp = new char[128];
        for (int i = 0; i < n; ++i) {
            mp[t.charAt(i)]++;
        }
        while (end < m) {
            if (mp[s.charAt(end++)]-- > 0)
                cnt--;
            while (cnt == 0) {
                if (end - begin < length) {
                    length = end - begin;
                    start = begin;
                }
                if (++mp[s.charAt(begin++)] > 0)
                    cnt++;
            }
        }
        return length == Integer.MAX_VALUE ? "" : s.substring(start, length + start);

    }

    public boolean validWordSquare(List<String> words) {
        int n = words.size();
        for (int i = 0; i < n; ++i) {
            int m = words.get(i).length();
            if (m > n)
                return false;
            for (int j = 0; j < m && j < n / 2; ++j) {
                if (words.get(i).charAt(j) != words.get(j).charAt(i))
                    return false;
            }
        }
        return true;
    }

    //481 magic string
    //using string would lead to LTE, so you would better use StringBuilder
    public int magicalString(int n) {
        if (n <= 0)
            return 0;
        StringBuilder sb = new StringBuilder("122");
        StringBuilder count = new StringBuilder("12");
        int index = 2, cnt = 0;
        while (sb.length() < n) {
            count = sb;
            int i = index;
            while (i < count.length()) {
                if (count.charAt(i) == '1') {
                    sb.append(sb.charAt(sb.length() - 1) == '2' ? '1' : '2');
                } else {
                    sb.append(sb.charAt(sb.length() - 1) == '2' ? "11" : "22");
                }
                if (sb.length() >= n)
                    break;
                i++;
            }
            index = count.length();
        }
        for (int i = 0; i < n; ++i) {
            if (sb.charAt(i) == '1')
                cnt++;
        }
        //System.out.println(s);
        return cnt;
    }






}
