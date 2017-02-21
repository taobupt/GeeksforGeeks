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
        FenwickTree fenwickTree = new FenwickTree(100050);
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
            int x = 100000;
            long kkk = fenwickTree.sum(x);
            long k2 = fenwickTree.sum(val1[i]);
            res += kkk - k2;
            fenwickTree.update(val[i], 1);
        }
        return new ArrayList<Integer>();
    }

    public int reversePairs(int[] nums) {
        FenwickTree fenwickTree = new FenwickTree(200050);
        List<Long> cnt = new ArrayList<>();
        int n = nums.length;
        for (int i = 0; i < n; ++i) {
            cnt.add((long) nums[i]);
            cnt.add(2 * (long) nums[i]);
        }
        Collections.sort(cnt);
        Map<Long, Integer> map = new HashMap<>();
        for (int i = 0; i < 2 * n; ++i) {
            map.put(cnt.get(i), i + 1);
        }
        int res = 0;
        for (int i = 0; i < n; ++i) {
            int x = 200000;
            long ceil = fenwickTree.sum(x);
            long floor = fenwickTree.sum(map.get(2 * (long) nums[i]));
            res += ceil - floor;
            fenwickTree.update(map.get((long) nums[i]), 1);
        }
        return res;
    }

    public void merge(int[] nums, int start, int mid, int end, int[] count) {
        for (int i = start, j = mid + 1; i <= mid; i++) {
            while (j <= end && nums[i] / 2.0 > nums[j]) j++;
            count[0] += j - (mid + 1);
        }
        int i = start, j = mid + 1, index = 0;
        int[] res = new int[end - start + 1];
        while (i <= mid && j <= end) {
            if (nums[i] > nums[j]) {
                res[index++] = nums[j++];
            } else
                res[index++] = nums[i++];

        }
        while (i <= mid) {
            res[index++] = nums[i++];
        }
        while (j <= end)
            res[index++] = nums[j++];
        for (int k = 0; k < res.length; ++k)
            nums[start + k] = res[k];
    }

    public void mergeSort(int[] nums, int start, int end, int[] count) {
        if (start >= end)
            return;
        int mid = (end - start) / 2 + start;
        mergeSort(nums, start, mid, count);
        mergeSort(nums, mid + 1, end, count);
        merge(nums, start, mid, end, count);
    }

    public void mergeSort(int[] nums) {
        int[] cnt = new int[1];
        mergeSort(nums, 0, nums.length - 1, cnt);
        System.out.println(cnt[0]);

    }

    public void dfs(int pos, int cnt[], int N, boolean[] vis) {
        if (pos == N + 1) {
            cnt[0]++;
            return;
        }
        for (int i = 1; i <= N; ++i) {
            if (!vis[i] && ((pos % i == 0) || (i % pos == 0))) {
                vis[i] = true;
                dfs(pos + 1, cnt, N, vis);
                vis[i] = false;
            }
        }
    }

    public int countArrangement(int N) {
        int[] cnt = new int[1];
        boolean[] vis = new boolean[20];
        dfs(1, cnt, N, vis);
        return cnt[0];
    }



}
