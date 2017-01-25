package hashTable;

import java.util.*;

/**
 * Created by taobupt on 1/17/2017.
 */

//295 find median from data stream

class MedianFinder {

    PriorityQueue<Integer> pq1 = null;//large elements
    PriorityQueue<Integer> pq2 = null;//small elements
    // Adds a number into the data structure.

    public MedianFinder() {
        pq2 = new PriorityQueue<>();
        pq1 = new PriorityQueue<>(Collections.reverseOrder());
    }

    public void addNum(int num) {
        if (pq2.isEmpty() || num >= pq2.peek())
            pq2.add(num);
        else
            pq1.add(num);
        while (pq2.size() < pq1.size()) {
            pq2.add(pq1.poll());
        }
        while (pq2.size() > pq1.size() + 1) {
            pq1.add(pq2.poll());
        }
    }

    // Returns the median of current data stream
    public double findMedian() {
        int size = pq1.size() + pq2.size();
        return size % 2 == 0 ? (pq1.peek() / 2.0 + pq2.peek() / 2.0) : 1.0 * pq2.peek();
    }
};


public class HASHTABLE {

    //477 number of boomeranges
    public int getDistance(int[] x1, int[] x2) {
        return (x1[0] - x2[0]) * (x1[0] - x2[0]) + (x1[1] - x2[1]) * (x1[1] - x2[1]);
    }

    public int numberOfBoomerangs(int[][] points) {
        List<Map<Integer, Integer>> li = new ArrayList<>();
        int n = points.length;
        for (int i = 0; i < n; ++i) {
            Map<Integer, Integer> map = new HashMap<>();
            for (int j = 0; j < n; ++j) {
                int dist = getDistance(points[i], points[j]);
                if (map.containsKey(dist)) {
                    map.put(dist, map.get(dist) + 1);
                } else
                    map.put(dist, 1);
            }
            li.add(map);
        }
        int cnt = 0;
        for (int i = 0; i < li.size(); ++i) {
            for (Map.Entry<Integer, Integer> entry : li.get(i).entrySet()) {
                int val = entry.getValue();
                if (val >= 2) {
                    cnt += (val - 1) * val;
                }
            }
        }
        return cnt;
    }

    //290 word pattern
    public boolean wordPattern(String pattern, String str) {
        String[] strs = str.split(" ");
        Map<String, Character> s2c = new HashMap<>();
        Map<Character, String> c2s = new HashMap<>();
        if (pattern.length() != strs.length)
            return false;
        int n = strs.length;
        for (int i = 0; i < n; ++i) {
            boolean existString = s2c.containsKey(strs[i]);
            boolean existCharacter = c2s.containsKey(pattern.charAt(i));
            if (!existCharacter && !existString) {
                s2c.put(strs[i], pattern.charAt(i));
                c2s.put(pattern.charAt(i), strs[i]);
                continue;
            }
            if ((existCharacter && !existString) || (!existCharacter && existString))
                return false;
            if (existCharacter && existString) {
                String sub = c2s.get(pattern.charAt(i));
                char c = s2c.get(strs[i]);
                if (!sub.equals(strs[i]) || c != pattern.charAt(i))
                    return false;
            }
        }
        return true;
    }

    //246 Strobogrammatic number
    public boolean isStrobogrammatic(String num) {
        Map<Character, Character> map = new HashMap<>();
        map.put('0', '0');
        map.put('1', '1');
        map.put('6', '9');
        map.put('8', '8');
        map.put('9', '6');
        StringBuilder sb = new StringBuilder(num);
        int n = num.length();
        for (int i = 0; i < n; ++i) {
            if (!map.containsKey(sb.charAt(i)))
                return false;
            else
                sb.setCharAt(i, map.get(sb.charAt(i)));
        }
        sb.reverse();
        return num.equals(sb.toString());
    }

    //two pointes is another way, just confirm whether they are equal
    public boolean isStrobogrammaticTwoPointers(String num) {
        Map<Character, Character> map = new HashMap<Character, Character>();
        map.put('6', '9');
        map.put('9', '6');
        map.put('0', '0');
        map.put('1', '1');
        map.put('8', '8');

        int l = 0, r = num.length() - 1;
        while (l <= r) {
            if (!map.containsKey(num.charAt(l))) return false;
            if (map.get(num.charAt(l)) != num.charAt(r))
                return false;
            l++;
            r--;
        }

        return true;
    }

    //247 strobogrammatic ii
    public List<String> findStrobogrammaticHelper(int n) {
        List<String> res = new ArrayList();
        if (n == 0) {
            res.add("");
            return res;
        }
        if (n == 1) {
            res.addAll(Arrays.asList(new String[]{"0", "1", "8"}));
            return res;
        } else {
            List<String> ans = findStrobogrammaticHelper(n - 2);
            for (String str : ans) {
                res.add("0" + str + "0");
                res.add("1" + str + "1");
                res.add("6" + str + "9");
                res.add("8" + str + "8");
                res.add("9" + str + "6");
            }
            return res;
        }
    }

    public List<String> findStrobogrammatic(int n) {
        List<String> ans = findStrobogrammaticHelper(n);
        //return ans;
        List<String> res = new ArrayList<>();
        for (String str : ans) {
            if (!str.startsWith("0") || (str.equals("0")))//n=1 is a special case
                res.add(str);
        }
        return res;
    }

    //another version

    public List<String> findStrobogrammaticConcise(int n) {
        return helper(n, n);//this is tricky
    }

    List<String> helper(int n, int m) {
        if (n == 0) return new ArrayList<String>(Arrays.asList(""));
        if (n == 1) return new ArrayList<String>(Arrays.asList("0", "1", "8"));

        List<String> list = helper(n - 2, m);

        List<String> res = new ArrayList<String>();

        for (int i = 0; i < list.size(); i++) {
            String s = list.get(i);

            if (n != m) res.add("0" + s + "0");

            res.add("1" + s + "1");
            res.add("6" + s + "9");
            res.add("8" + s + "8");
            res.add("9" + s + "6");
        }

        return res;
    }

    //248 strobogrammaticInRange
    //I get the rule, but I can't calculate the number of string who is larger the lower in the same level






}
