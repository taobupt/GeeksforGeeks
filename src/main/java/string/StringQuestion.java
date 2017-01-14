package string;

import java.util.HashMap;
import java.util.Map;
import java.util.Stack;

/**
 * Created by Tao on 1/13/2017.
 */
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


}
