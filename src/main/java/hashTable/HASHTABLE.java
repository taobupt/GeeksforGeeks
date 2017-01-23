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


}
