package mathQuestion;

import java.util.*;

/**
 * Created by taobupt on 1/17/2017.
 */
public class MathQuestion {
    //396
    //rotate function
    //conner case []
    //-2147483648,-2147483648
    public int maxRotateFunction(int[] A) {
        int sum1 = 0, res = Integer.MIN_VALUE, aSum = 0;
        int n = A.length;
        if (n <= 1)
            return 0;
        for (int i = 0; i < n; ++i) {
            sum1 += i * A[i];
            aSum += A[i];
        }
        for (int i = 0; i < n; ++i) {

            res = Math.max(res, sum1);
            sum1 += aSum - n * A[n - i - 1];
        }
        return res;
    }

    //356 line reflection
    //potential line is double
    //same points
    public boolean isReflected(int[][] points) {
        int minX = Integer.MAX_VALUE, maxX = Integer.MIN_VALUE;
        int n = points.length;
        int same = 0;
        Map<Double, ArrayList<Integer>> map = new HashMap<>();
        double potentialLine = 0;
        for (int i = 0; i < n; ++i) {
            minX = Math.min(minX, points[i][0]);
            maxX = Math.max(maxX, points[i][0]);
            if (map.containsKey(1.0 * points[i][0])) {
                if (map.get(1.0 * points[i][0]).contains(points[i][1]))
                    same++;
                map.get(1.0 * points[i][0]).add(points[i][1]);
            } else {
                ArrayList<Integer> tmp = new ArrayList<>();
                tmp.add(points[i][1]);
                map.put(points[i][0] * 1.0, tmp);
            }
        }
        potentialLine = (maxX + minX) / 2.0;
        int onTheLine = 0;
        int leftcnt = 0;
        Map<Integer, Integer> vis = new HashMap<>();
        for (int i = 0; i < n; ++i) {
            if (vis.containsKey(points[i][0]) && vis.get(points[i][0]) == points[i][1])
                continue;
            vis.put(points[i][0], points[i][1]);
            if (1.0 * points[i][0] < potentialLine) {
                leftcnt++;
                double reflectionx = 2 * potentialLine - points[i][0];
                if (map.containsKey(reflectionx) && map.get(reflectionx).contains(points[i][1])) {
                    if (map.get(reflectionx).size() == 1)
                        map.remove(reflectionx);
                    else {
                        int ind = map.get(reflectionx).indexOf(points[i][1]);
                        int val = map.get(reflectionx).remove(ind);
                    }

                } else
                    return false;

            } else if (1.0 * points[i][0] == potentialLine) {
                onTheLine++;
            }
        }
        return 2 * leftcnt == (n - onTheLine - same);
    }

    //revised
    public boolean isReflectedConcise(int[][] points) {
        int max = Integer.MIN_VALUE;
        int min = Integer.MAX_VALUE;
        Set<String> set = new HashSet<>();
        for (int[] p : points) {
            max = Math.max(max, p[0]);
            min = Math.min(min, p[0]);
            String str = p[0] + "a" + p[1];
            set.add(str);
        }
        int sum = max + min;//potential line;
        for (int[] p : points) {
            //check every points
            String str = (sum - p[0]) + "a" + p[1];
            if (!set.contains(str))
                return false;
        }
        return true;
    }

    //Minimum Moves to Equal Array Elements
    //sum+(n-1)*m=x*n;
    //x=min+m;
    //sum-m=min*n;
    //m=sum-min*n;
    public long minMoves(int[] nums) {
        long res = 0;
        int minValue = Integer.MIN_VALUE, n = nums.length;
        for (int x : nums) {
            minValue = Math.min(minValue, x);
            res += x;
        }
        return res - n * minValue;
    }

    //462 minimum moves to equal array elementsII
    public int minMoves2(int[] nums) {
        Arrays.sort(nums);
        int begin = 0, end = nums.length - 1, cnt = 0;
        while (begin < end) {
            cnt += nums[end--] - nums[begin++];
        }
        return cnt;
    }

    //you can find the middle number
    public int minTotalDistance(int[][] grid) {
        List<Integer> xx = new ArrayList<>();
        List<Integer> yy = new ArrayList<>();
        if (grid.length == 0 || grid[0].length == 0)
            return 0;
        int m = grid.length, n = grid[0].length;
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (grid[i][j] == 1) {
                    xx.add(i);
                    yy.add(j);
                }
            }
        }
        Collections.sort(xx);
        Collections.sort(yy);
        int cnt = 0, begin = 0, end = xx.size() - 1;
        while (begin < end) {
            cnt += xx.get(end) - xx.get(begin) + yy.get(end) - yy.get(begin);
        }
        return cnt;

    }


    //386 386. Lexicographical Numbers
    public void dfs(int n, List<Integer> res, int index) {
        if (index <= n) {
            res.add(index);
        } else if (index > n)
            return;
        for (int j = 0; j <= 9; ++j) {
            if (10 * index + j <= n)
                dfs(n, res, 10 * index + j);
            else
                return;
        }
    }

    public List<Integer> lexicalOrder(int n) {
        List<Integer> res = new ArrayList<>();
        for (int i = 1; i <= 9; ++i)
            dfs(n, res, i);
        return res;
    }

    //440 K-th Smallest in Lexicographical Order
    //move (k - 1) steps to the target node
    public int findKthNumber(int n, int k) {
        int curr = 1;
        k = k - 1;
        while (k > 0) {
            int steps = calSteps(n, curr, curr + 1);
            if (steps <= k) {
                curr += 1;
                k -= steps;
            } else {
                curr *= 10;
                k -= 1;
            }
        }
        return curr;
    }

    public int calSteps(int n, long n1, long n2) {
        int steps = 0;
        while (n1 <= n) {
            steps += Math.min(n + 1, n2) - n1;//到达n+1之后才能算是跳了n步
            n1 *= 10;
            n2 *= 10;
        }
        return steps;
    }

}
