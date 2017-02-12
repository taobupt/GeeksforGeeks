package leetcodeContest;

import java.util.*;

/**
 * Created by Tao on 2/11/2017.
 */

class FenwickTree {
    public int n;
    public long[] sumArray = null;

    public long lowbit(long x) {
        return x & (-x);
    }

    public FenwickTree(int n) {
        this.n = n;
        sumArray = new long[n + 1];
    }

    public void update(int x, long d) {
        while (x <= n) {
            sumArray[x] += d;
            x += lowbit(x);
        }
    }

    public long sum(int x) {
        long sum = 0;
        while (x > 0) {
            sum += sumArray[x];
            x -= lowbit(x);
        }
        return sum;
    }

}

public class ContestQuestion {
    //contest 19 2/11/2017
    public String convertTo7(int num) {
        StringBuilder sb = new StringBuilder("");
        boolean negative = num >= 0 ? false : true;
        int numabs = Math.abs(num);
        while (numabs > 0) {
            sb.append(numabs % 7);
            numabs /= 7;
        }
        if (negative)
            sb.append('-');
        sb.reverse();
        return sb.toString();

    }

    public int lowerBound(List<Long> nums, long target) {
        int index = 0, begin = 0, end = nums.size() - 1;
        if (nums.get(end) < target)
            return end + 1;
        while (begin < end) {
            int mid = (end - begin) / 2 + begin;
            if (nums.get(mid) >= target) {
                end = mid;
                index = end;
            } else {
                begin = mid + 1;
                index = begin;
            }
        }
        return index;
    }

    public List<Integer> countSmaller(int[] nums) {
        FenwickTree fenwickTree = new FenwickTree(200050);
        List<Long> cnt = new ArrayList<>();
        int n = nums.length;
        for (int i = 0; i < n; ++i) {
            cnt.add((long) nums[i]);
            cnt.add(2 * (long) nums[i]);
        }
        Collections.sort(cnt);
        int[] val = new int[n];
        for (int i = 0; i < n; ++i) {
            val[i] = lowerBound(cnt, nums[i]);
            val[i]++;
        }
        int[] val1 = new int[n];
        for (int i = 0; i < n; ++i) {
            val1[i] = lowerBound(cnt, 2 * (long) nums[i]);
            val1[i]++;
        }
        int res = 0;
        for (int i = 0; i < n; ++i) {
            int x = 200000;
            long kkk = fenwickTree.sum(x);
            long k2 = fenwickTree.sum(val1[i]);
            res += kkk - k2;
            fenwickTree.update(val[i], 1);
        }
        return new ArrayList<Integer>();
    }
}
