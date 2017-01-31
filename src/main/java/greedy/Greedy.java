package greedy;

import java.util.ArrayList;
import java.util.Comparator;
import java.util.List;
import java.util.PriorityQueue;

import common.SpecialInterval;
import common.Interval;


/**
 * Created by Tao on 1/30/2017.
 */


public class Greedy {

    //484 find permutations
    public int[] findPermutation(String s) {
        int n = s.length();
        int[] res = new int[n + 1];
        PriorityQueue<Integer> pq = new PriorityQueue<>();
        for (int i = 1; i <= n + 1; ++i)
            pq.offer(i);
        int j = 0, count = 1, index = 0;
        while (j < n) {
            if (s.charAt(j) == 'D') {
                while (j < n - 1 && s.charAt(j) == s.charAt(j + 1)) {
                    count++;
                    j++;
                }
                int[] tmp = new int[count + 1];
                for (int i = 0; i <= count; ++i) {
                    tmp[i] = pq.poll();
                }
                while (count >= 1)
                    res[index++] = tmp[count--];
                pq.offer(tmp[0]);
                count = 1;
            } else {
                while (j < n - 1 && s.charAt(j) == s.charAt(j + 1)) {
                    count++;
                    j++;
                }
                for (int i = 0; i < count; ++i) {
                    res[index++] = pq.poll();
                }
                count = 1;
            }
            j++;
        }
        if (!pq.isEmpty())
            res[index] = pq.poll();
        return res;
    }

    //452. Minimum Number of Arrows to Burst Balloons
    public int findMinArrowShots(int[][] points) {
        List<Interval> res = new ArrayList<>();
        int n = points.length;
        for (int[] point : points)
            res.add(new Interval(point[0], point[1]));
        res.sort(new Comparator<Interval>() {
            @Override
            public int compare(Interval o1, Interval o2) {
                if (o1.start < o2.start || (o1.start == o2.start && o1.end < o2.end))
                    return -1;
                else if (o1.start > o2.start || (o1.start == o2.start && o1.end > o2.end))
                    return 1;
                else
                    return 0;

            }
        });
        int cnt = 0, j = 0;
        while (j < n) {
            int minval = res.get(j).end;
            while (j < n && res.get(j).start <= minval) {
                j++;
                if (j < n)
                    minval = Math.min(minval, res.get(j).end);
            }

            cnt++;
        }
        return cnt;
    }
}
