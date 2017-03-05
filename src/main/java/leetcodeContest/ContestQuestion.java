package leetcodeContest;

import org.omg.CORBA.INTERNAL;
import sun.awt.image.IntegerInterleavedRaster;

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


    //02/05/2017 contest 21
    //problem 1


    public int binarySearch(List<Integer> sss, int index) {
        if (sss.isEmpty())
            return index;
        int begin = 0, end = sss.size() - 1;
        while (begin < end) {
            int mid = (end - begin) / 2 + begin;
            if (sss.get(mid) <= index)
                begin = mid + 1;
            else
                end = mid;
        }
        System.out.println(sss.get(begin));
        return sss.get(begin) > index ? sss.get(begin) : -1;
    }

    public String findLongestWord(String s, List<String> d) {
        List<List<Integer>> cnt = new ArrayList<>();
        for (int i = 0; i < 26; ++i) {
            cnt.add(new ArrayList<>());
        }
        int n = s.length();
        String ans = "";
        for (int i = 0; i < n; ++i)
            cnt.get((int) (s.charAt(i) - 'a')).add(i);
        for (String word : d) {
            int nn = word.length();
            int ind = -1;
            int i = 0;
            for (; i < nn; ++i) {
                int xxx = binarySearch(cnt.get(word.charAt(i) - 'a'), ind);
                if (xxx > ind)
                    ind = xxx;
                else
                    break;
            }
            if (i == nn) {
                if (ans.length() < word.length())
                    ans = word;
                else if (ans.length() == word.length()) {
                    int nnn = word.length();
                    for (int ii = 0; ii < nnn; ++ii) {
                        if (word.charAt(ii) > ans.charAt(ii)) {
                            break;
                        } else if (word.charAt(ii) < ans.charAt(ii)) {
                            ans = word;
                            break;
                        }
                    }
                }

            }
        }
        return ans;
    }


    public boolean checkSubarraySum(int[] nums, int k) {
        int n = nums.length;
        int[] sums = new int[n];
        sums[0] = nums[0];
        for (int i = 1; i < n; ++i)
            sums[i] = sums[i - 1] + nums[i];
        for (int i = 0; i < n; ++i) {
            if (sums[i] % k == 0)
                return true;
            for (int j = 0; j < i; ++j) {
                if ((sums[i] - sums[j]) % k == 0)
                    return true;
            }
        }
        return false;
    }


    //
    public int findPairs(int[] nums, int k) {
        int n = nums.length;
        Set<Integer> set = new HashSet<>();
        int cnt = 0;
        for (int x : nums)
            set.add(x);
        if (k == 0) {
            Map<Integer, Integer> map = new HashMap<>();
            for (int x : nums) {
                if (!map.containsKey(x))
                    map.put(x, 1);
                else
                    map.put(x, map.get(x) + 1);
            }
            for (int x : nums) {
                if (map.get(x) > 1)
                    cnt++;
            }
            return cnt;
        }
        if (k < 0)
            return 0;
        List<Integer> ans = new ArrayList<>();
        ans.addAll(set);

        Collections.sort(ans);
        n = ans.size();
        for (int i = 0; i < n; ++i) {
            if (set.contains(ans.get(i) + k))
                cnt++;
        }
        return cnt;
    }

    public int findLonelyPixel(char[][] picture) {
        if (picture.length == 0 || picture[0].length == 0)
            return 0;
        int m = picture.length, n = picture[0].length;
        int[] rows = new int[m];
        int[] cols = new int[n];
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (picture[i][j] == 'B')
                    rows[i]++;
            }
        }


        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                if (picture[j][i] == 'B')
                    cols[i]++;
            }
        }

        int cnt = 0;
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (picture[i][j] == 'B') {
                    if (rows[i] == 1 && cols[j] == 1)
                        cnt++;
                }
            }
        }
        return cnt;
    }

    public int findBlackPixelII(char[][] picture, int N) {
        if (picture.length == 0 || picture[0].length == 0)
            return 0;
        int m = picture.length, n = picture[0].length;
        int[] rows = new int[m];
        int[] cols = new int[n];
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (picture[i][j] == 'B')
                    rows[i]++;
            }
        }


        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < m; ++j) {
                if (picture[j][i] == 'B')
                    cols[i]++;
            }
        }

        int cnt = 0;
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (picture[i][j] == 'B') {
                    if (rows[i] == N && cols[j] == N) {
                        boolean flag = true;
                        for (int k = 0; k < m; ++k) {
                            if (k == i)
                                continue;
                            if (picture[k][j] == 'B') {
                                for (int z = 0; z < n; ++z) {
                                    if (picture[k][z] != picture[i][z]) {
                                        flag = false;
                                        break;
                                    }
                                }
                            }
                            if (!flag)
                                break;
                        }
                        if (flag)
                            cnt++;
                    }

                }
            }
        }
        return cnt;
    }


    public int indexof(char[] arrs, char c, boolean isshun) {
        int cnt = 0, n = arrs.length;
        char[] cl = arrs.clone();
        if (isshun) {
            for (int i = n - 1; i >= 0; --i) {
                if (arrs[i] == c)
                    break;
                else
                    cnt++;
            }

            for (int i = 0; i < n; ++i) {
                arrs[i] = cl[(i + cnt) % n];
            }
        } else {
            for (int i = 0; i < n; ++i) {
                if (arrs[i] == c)
                    break;
                else
                    cnt++;
            }
            for (int i = 0; i < n; ++i) {
                arrs[i] = cl[(i - cnt + n) % n];
            }
        }
        return cnt;
    }

    public int findRotateSteps(String ring, String key) {
        //无非就是两种方法，顺时针和逆时针

        char[] keys = key.toCharArray();
        char[] rings = ring.toCharArray();
        char[] ringshun = rings.clone();
        char[] ringsni = rings.clone();
        int m = key.length();
        int dp[][] = new int[m + 1][2];//正反两方向
        for (int i = 1; i <= m; ++i) {
            int cntshun = indexof(ringshun, keys[i - 1], true);
            int cntni = indexof(ringsni, keys[i - 1], false);
            dp[i][1] = Math.min(dp[i - 1][0], dp[i - 1][1]) + cntni;
            dp[i][0] = Math.min(dp[i - 1][0], dp[i - 1][1]) + cntshun;
        }

        return Math.min(dp[m][0], dp[m][1]);
    }

    //another version
    public int findRotateStepsDP(String ring, String key) {
        int m = ring.length(), n = key.length();
        int[][] dp = new int[n + 1][m];
        for (int i = 0; i <= n; ++i) {
            for (int j = 0; j < m; ++j)
                dp[i][j] = Integer.MAX_VALUE;
        }
        for (int i = 0; i < m; ++i)
            dp[n][i] = 0;
        int dist = 0;
        for (int i = n - 1; i >= 0; --i) {
            for (int j = 0; j < m; ++j) {
                for (int k = 0; k < m; ++k) {
                    if (ring.charAt(k) != key.charAt(i))
                        continue;
                    dist = Math.min((k - j + m) % m, (j - k + m) % m);
                    dp[i][j] = Math.min(dp[i][j], dp[i + 1][k] + dist);

                }
            }
        }
        return dp[0][0] + n;
    }











}
