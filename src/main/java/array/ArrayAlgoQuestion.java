package array;


import common.Project;

import java.util.*;

/**
 * Created by tao on 1/9/17.
 */

class NumMatrix {

    public int[][] sumArray = null;
    public int[][] matrix = null;
    public int m = 0;
    public int n = 0;

    public int lowbit(int x) {
        return x & (-x);
    }

    public NumMatrix(int[][] matrix) {
        this.matrix = matrix;
        if (matrix.length != 0) {
            this.m = matrix.length;
            this.n = matrix[0].length;
        }
        sumArray = new int[m + 1][n + 1];
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                add(i + 1, j + 1, matrix[i][j]);
            }
        }
    }

    public void add(int x, int y, int val) {
        for (int i = x; i <= m; i += lowbit(i)) {
            for (int j = y; j <= n; j += lowbit(j)) {
                sumArray[i][j] += val;
            }
        }
    }


    public int sum(int x, int y) {
        int res = 0;
        for (int i = x; i > 0; i -= lowbit(i)) {
            for (int j = y; j > 0; j -= lowbit(j))
                res += sumArray[i][j];
        }
        return res;
    }

    public void update(int row, int col, int val) {
        add(row + 1, col + 1, val - matrix[row][col]);
        add(row + 1, col, val - matrix[row][col]);
        add(row, col + 1, val - matrix[row][col]);
        matrix[row][col] = val;
    }

    public int sumRegion(int row1, int col1, int row2, int col2) {
        return m == 0 || n == 0 ? 0 : sum(row2 + 1, col2 + 1) + sum(row1, col1) - sum(row2 + 1, col1) - sum(row1, col2 + 1);
    }
}


//307 range sum query mutable
class NumArray {
    public int[] sumArray = null;
    public int n = 0;
    public int[] nums = null;

    public int lowbit(int x) {
        return x & (-x);
    }


    public void add(int x, int val) {
        while (x <= n) {
            sumArray[x] += val;
            x += lowbit(x);
        }
    }

    public int sum(int x) {
        int res = 0;
        while (x > 0) {
            res += sumArray[x];
            x -= lowbit(x);
        }
        return res;
    }

    public NumArray(int[] nums) {
        this.n = nums.length;
        sumArray = new int[n + 1];
        this.nums = nums;
        for (int i = 0; i < n; ++i)
            add(i + 1, nums[i]);
    }

    public void update(int i, int val) {
        add(i + 1, val - nums[i]);
        nums[i] = val;//this can influence the above instructions
    }

    public int sumRange(int i, int j) {
        return n == 0 ? 0 : sum(j + 1) - sum(i);//from 1
    }
}



class SummaryRanges {

    /**
     * Initialize your data structure here.
     */
    public List<Interval> intervals = null;

    public SummaryRanges() {
        intervals = new ArrayList<>();
    }

    public void addNum(int val) {
        Interval newInterval = new Interval(val, val);
        List<Interval> res = new ArrayList<>();
        int i = 0, n = intervals.size();
        while (i < n) {
            if (intervals.get(i).end < newInterval.start - 1)
                res.add(intervals.get(i++));
            else
                break;
        }
        while (i < n && intervals.get(i).start <= newInterval.end + 1) {
            newInterval.start = Math.min(intervals.get(i).start, newInterval.start);
            newInterval.end = Math.max(intervals.get(i).end, newInterval.end);
            i++;
        }
        res.add(newInterval);
        while (i < n) {
            res.add(intervals.get(i++));
        }
        intervals = res;
    }

    public List<Interval> getIntervals() {
        return intervals;
    }
}

class Unit {
    Interval a;
    int index;

    Unit(Interval a, int index) {
        this.a = a;
        this.index = index;
    }
}

class Interval {
    int start;
    int end;

    Interval(int s, int e) {
        this.start = s;
        this.end = e;
    }

    Interval() {
        this.start = 0;
        this.end = 0;
    }

    public String toString() {
        return "[" + start + " " + end+"]";
    }
}

class RandomizedSet {


    private Map<Integer, Integer> mp = null;
    private List<Integer> nums = null;
    private Random rand;

    /**
     * Initialize your data structure here.
     */
    public RandomizedSet() {
        nums = new ArrayList<Integer>();
        mp = new HashMap<Integer, Integer>();
        rand = new Random();
    }

    /**
     * Inserts a value to the set. Returns true if the set did not already contain the specified element.
     */
    public boolean insert(int val) {
        if (mp.containsKey(val))
            return false;
        mp.put(val, nums.size());
        nums.add(val);
        return true;
    }

    /**
     * Removes a value from the set. Returns true if the set contained the specified element.
     */
    public boolean remove(int val) {
        if (!mp.containsKey(val))
            return false;
        //swap to last and delete
        //judge whether is the same value
        if (mp.get(val) < nums.size() - 1) {
            int last = nums.get(nums.size() - 1);
            nums.set(mp.get(val), last);
            mp.put(last, mp.get(val));
        }
        mp.remove(val);
        nums.remove(nums.size() - 1);
        return true;
    }

    /**
     * Get a random element from the set.
     */
    public int getRandom() {
        return nums.get(rand.nextInt(nums.size()));
    }
}


class WordDistance {

    private Map<String, List<Integer>> map = new HashMap<String, List<Integer>>();

    public WordDistance(String[] words) {
        int n = words.length;
        for (int i = 0; i < n; ++i) {
            if (!map.containsKey(words[i])) {
                List<Integer> index = new ArrayList<Integer>();
                index.add(i);
                map.put(words[i], index);
            } else
                map.get(words[i]).add(i);
        }
    }

    public int shortest(String word1, String word2) {
        List<Integer> index1 = map.get(word1);
        List<Integer> index2 = map.get(word2);
        int i = 0, j = 0, m = index1.size(), n = index2.size();
        int res = Integer.MAX_VALUE;
        while (i < m && j < n) {
            res = Math.min(res, Math.abs(index1.get(i) - index2.get(j)));
            if (index1.get(i) < index2.get(j))
                i++;
            else
                j++;
        }
        return res;
    }
}


class ValueWtihFrequency {
    public int val;
    public int count;
    public int index;

    public ValueWtihFrequency(int val, int count, int index) {
        this.val = val;
        this.count = count;
        this.index = index;
    }
}



public class ArrayAlgoQuestion {

    public ArrayAlgoQuestion() {

    }

    //169 majority elements
    //first use hashmap
    public int majorityElement(int[] nums) {
        Map<Integer, Integer> map = new HashMap<Integer, Integer>();
        int n = nums.length;
        for (int x : nums) {
            if (map.containsKey(x)) {
                map.put(x, map.get(x) + 1);
            } else {
                map.put(x, 1);
            }
            if (map.get(x) > n / 2)
                return x;
        }
        return -1;
    }

    //vote algorithms
    public int majorityElementBetter(int[] nums) {
        if (nums == null || nums.length == 0)
            return -1;
        int count = 1;
        int res = nums[0];
        for (int i = 1; i < nums.length; ++i) {
            if (count == 0) {
                count = 1;
                res = nums[i];
            } else if (nums[i] == res) {
                count++;
            } else if (nums[i] != res) {
                count--;
            }
        }
        return res;
    }

    //229 majority element II
    // compare a==nums[i] && b==nums[i] first and then compare the count=0
    //[8,8,7,7,7] is the case
    public List<Integer> majorityElementII(int[] nums) {
        List<Integer> res = new ArrayList<Integer>();
        if (nums == null || nums.length == 0)
            return res;
        int n = nums.length;
        int a = 0;
        int cnt_a = 0;
        int b = 0;
        int cnt_b = 0;
        for (int i = 0; i < n; ++i) {
            if (a == nums[i]) {
                cnt_a++;
            } else if (b == nums[i]) {
                cnt_b++;
            } else if (cnt_a == 0) {
                cnt_a++;
                a = nums[i];
            } else if (cnt_b == 0) {
                b = nums[i];
                cnt_b++;
            } else {
                cnt_a--;
                cnt_b--;
            }
        }
        cnt_a = 0;
        cnt_b = 0;
        for (int x : nums) {
            if (x == a)
                cnt_a++;
            else if (x == b)
                cnt_b++;
        }
        if (cnt_a > n / 3)
            res.add(a);
        if (cnt_b > n / 3)
            res.add(b);
        return res;
    }

    //extension
    //find all elements that appear more than [n/k]
    public List<Integer> kmajorityElements(int[] nums) {
        int n = nums.length, k = 3;//you can modify here
        List<Integer> res = new ArrayList<Integer>();
        if (n == 0)
            return res;
        int[] candidates = new int[k - 1];
        int[] counts = new int[k - 1];
        for (int num : nums) {
            boolean settled = false;
            for (int i = 0; i < k - 1; ++i) {
                if (candidates[i] == num) {
                    counts[i]++;
                    settled = true;
                    break;
                }
            }
            if (settled)
                continue;
            for (int i = 0; i < k - 1; ++i) {
                if (counts[i] == 0) {
                    counts[i] = 1;
                    candidates[i] = num;
                    settled = true;
                    break;
                }
            }
            if (settled)
                continue;
            for (int i = 0; i < k - 1; ++i) {
                counts[i] = (counts[i] > 0) ? (counts[i] - 1) : 0;
            }
        }
        //check the value;
        Arrays.fill(counts, 0);
        for (int num : nums) {
            for (int i = 0; i < k - 1; ++i) {
                if (candidates[i] == num) {
                    counts[i]++;
                    break;
                }
            }
        }
        for (int i = 0; i < k - 1; ++i)
            if (counts[i] > n / k)
                res.add(candidates[i]);
        return res;
    }

    //1 two sum
    //use two pointers
    public int[] twoSum(int[] nums, int target) {
        Map<Integer, Integer> map = new HashMap<Integer, Integer>();//store element in the array and its index
        int n = nums.length;
        int[] res = {0, 0};
        for (int i = 0; i < n; ++i) {
            if (map.containsKey(target - nums[i])) {
                res[0] = map.get(nums[i]);
                res[1] = i;
                break;
            }
            map.put(nums[i], i);
        }
        return res;
    }

    //121 best time to buy and sell stock
    // [1,7,4,11]; if you change it into [0,6,-3,7], then you will get the maximum subarray sum problem
    public int maxProfit(int[] prices) {
        int minValue = Integer.MAX_VALUE;
        int res = 0;
        for (int price : prices) {
            minValue = Math.min(minValue, price);
            res = Math.max(res, price - minValue);
        }
        return res;
    }

    //53 maximum subarray
    //kadane algo
    //if wan you to calculate the start point and end point
    public int maxSubArray(int[] nums) {
        int sum = 0, n = nums.length, start = 0, end = 0, s = 0;
        int res = Integer.MIN_VALUE;
        for (int i = 0; i < n; ++i) {
            sum += nums[i];
            if (sum > res) {
                res = sum;
                start = s;
                end = i;
            }
            if (sum < 0) {
                sum = 0;
                s = i + 1;
            }
        }
        return res;
    }

    //dynamic programming
    //O(N) space
    public int maxSubarrayDp(int[] nums) {
        int n = nums.length;
        int[] dp = new int[n];
        if (n == 0)
            return Integer.MIN_VALUE;
        int res = dp[0] = nums[0];
        for (int i = 1; i < n; ++i) {
            dp[i] = Math.max(dp[i - 1] + nums[i], nums[i]);
            res = Math.max(res, dp[i]);
        }
        return res;
    }

    //divide and conquer solution
    //mx (largest sum of this subarray),
    //lmx(largest sum starting from the left most element),
    //rmx(largest sum ending with the right most element),
    //sum(the sum of the total subarray).
    //the recurrence is T(n) = 2T(n / 2) + O(1). So the running time of this algorithm is O(n).
    //mx = max(max(mx1, mx2), rmx1 + lmx2);
    //lmx = max(lmx1, sum1 + lmx2);
    //rmx = max(rmx2, sum2 + rmx1);
    //sum = sum1 + sum2;
    public void maxSubArray(int[] nums, int l, int r, int[] sums) {
        if (l == r) {
            sums[0] = sums[1] = sums[2] = sums[2] = nums[l];
        } else {
            int m = l + (r - l) / 2;
            int[] sums1 = {0, 0, 0, 0};
            int[] sums2 = {0, 0, 0, 0};
            maxSubArray(nums, l, m, sums1);
            maxSubArray(nums, m + 1, r, sums2);
            sums[3] = Math.max(Math.max(sums1[3], sums2[3]), sums1[1] + sums2[1]);
            sums[2] = sums1[2] + sums2[2];
            sums[1] = Math.max(sums2[1], sums2[2] + sums1[1]);
            sums[0] = Math.max(sums1[0], sums1[2] + sums2[0]);
        }
    }

    public int maxSubArrayDC(int[] nums) {
        int n = nums.length;
        if (n == 0)
            return 0;
        int[] sums = {0, 0, 0, 0};//lmx,rmx,sum,mx;
        maxSubArray(nums, 0, n - 1, sums);
        return sums[3];
    }

    //another understandable divide and conquer solution
    public int maxSubarray(int[] nums) {
    /*
    Divide-and-conquer method.
    The maximum summation of subarray can only exists under following conditions:
    1. the maximum summation of subarray exists in left half.
    2. the maximum summation of subarray exists in right half.
    3. the maximum summation of subarray exists crossing the midpoints to left and right.
    1 and 2 can be reached by using recursive calls to left half and right half of the subarraies.
    Condition 3 can be found starting from the middle point to the left,
    then starting from the middle point to the right. Then adds up these two parts and return.

    T(n) = 2*T(n/2) + O(n)
    this program runs in O(nlogn) time
    */
        int maxsum = subArray(nums, 0, nums.length - 1);
        return maxsum;
    }

    public int subArray(int[] A, int left, int right) {
        if (left == right) {
            return A[left];
        }
        int mid = left + (right - left) / 2;
        int leftsum = subArray(A, left, mid);
        int rightsum = subArray(A, mid + 1, right);
        int middlesum = midSubArry(A, left, mid, right);
        return Math.max(leftsum, Math.max(rightsum, middlesum));
    }

    public int midSubArry(int[] A, int left, int mid, int right) {
        int leftsum = Integer.MIN_VALUE;
        int rightsum = Integer.MIN_VALUE;
        int sum = 0;
        for (int i = mid; i >= left; --i) {
            sum += A[i];
            if (leftsum < sum) {
                leftsum = sum;
            }
        }
        sum = 0;
        for (int j = mid + 1; j <= right; ++j) {
            sum += A[j];
            if (rightsum < sum) {
                rightsum = sum;
            }
        }
        return leftsum + rightsum;
    }

    //243 shortest word distance
    //best solution
    //otherwise, you can use vector to store the index first
    //then do two loops to compare the shortest index
    public int shortestDistance(String[] words, String word1, String word2) {
        int n = words.length;
        if (n < 2)
            return Integer.MAX_VALUE;
        int idx1 = -1, idx2 = -1, shortestIdx = Integer.MAX_VALUE;
        for (int i = 0; i < n; ++i) {
            if (words[i].equals(word1)) {
                idx1 = i;
            } else if (words[i].equals(word2)) {
                idx2 = i;
            }
            if (idx1 != -1 && idx2 != -1) {
                shortestIdx = Math.min(shortestIdx, Math.abs(idx1 - idx2));
            }
        }
        return shortestIdx;
    }

    //use only one variable

    public int shortestDistanceOneVariable(String[] words, String word1, String word2) {
        int index = -1;
        int n = words.length;
        int minDistance = Integer.MAX_VALUE;
        for (int i = 0; i < n; ++i) {
            if (words[i].equals(word1) || words[i].equals(word2)) {
                if (index != -1 && !words[index].equals(words[i])) {
                    minDistance = Math.min(minDistance, i - index);
                }
                index = i;
            }
        }
        return minDistance;
    }

    //Find the minimum distance between two numbers
    public int minDist(int[] nums, int x, int y) {
        int index = -1, n = nums.length, minD = Integer.MAX_VALUE;
        for (int i = 0; i < n; ++i) {
            if (nums[i] == x || nums[i] == y) {
                if (index != -1 && nums[index] != nums[i]) {
                    minD = Math.min(minD, i - index);
                }
                index = i;
            }
        }
        return minD;
    }

    //or you can get two index array first and then you can start to use two pointers


    //245 shortest word distanceII
    public int shortestWordDistance(String[] words, String word1, String word2) {
        int index = -1;
        int n = words.length;
        int minDistance = Integer.MAX_VALUE;
        for (int i = 0; i < n; ++i) {
            if (words[i].equals(word1) || words[i].equals(word2)) {
                if (index != -1 && (!words[index].equals(words[i]) || word1.equals(word2))) {
                    minDistance = Math.min(minDistance, i - index);
                }
                index = i;
            }
        }
        return minDistance;
    }

    //also you can split into part, equals and not equals are not together

    //244 shortest word distanceII



    //move zeros 283
    public void moveZeroes(int[] nums) {
        int i = 0, j = 0, n = nums.length;
        while (i < n) {
            if (nums[i] != 0)
                nums[j++] = nums[i];
            i++;
        }
        while (j < n)
            nums[j++] = 0;

    }

    //minimize the total number of operations
    //when the order is not important
    public void moveZerosBetter(int[] nums) {
        int i = 0, j = nums.length - 1;
        while (i < j) {
            while (i < j && nums[i] != 0)
                i++;
            while (i < j && nums[j] == 0)
                j--;
            int tmp = nums[j];
            nums[j--] = nums[i];
            nums[i++] = tmp;
        }
    }

    //Tecent summer intern question
    // keep order you should swap

    //the above is not we want
    public String moveCharacterBetter(String nums) {
        StringBuilder sb = new StringBuilder(nums);
        int d = 0, n = nums.length();
        for (int i = 0; i < n; ++i) {
            char c = nums.charAt(i);
            if (c <= 90 && c >= 65) {
                sb.deleteCharAt(i - d++).append(c);
            }
        }
        return sb.toString();
    }

    //414 third maximum number
    public int thirdMax(int[] nums) {
        if (nums == null || nums.length == 0)
            return Integer.MIN_VALUE;
        int n = nums.length;
        long firstMax = Long.MIN_VALUE;
        long secMax = Long.MIN_VALUE;
        long thirdMax = Long.MIN_VALUE;
        for (int x : nums) {
            if (x == firstMax || x == secMax)
                continue;
            if (x > firstMax) {
                thirdMax = secMax;
                secMax = firstMax;
                firstMax = x;
            } else if (x > secMax) {
                thirdMax = secMax;
                secMax = x;
            } else if (x > thirdMax) {
                thirdMax = x;
            }
        }
        return thirdMax == Long.MIN_VALUE ? (int) firstMax : (int) thirdMax;
    }

    //use both count and value to check dups,no long.value
    public int thirdMaxConcise(int[] nums) {
        int count = 0, max, mid, small;
        max = mid = small = Integer.MIN_VALUE;
        for (int num : nums) {
            if (count > 0 && num == max || count > 1 && num == mid)
                continue;
            count++;
            if (num > max) {
                small = mid;
                mid = max;
                max = num;
            } else if (num > mid) {
                small = mid;
                mid = num;
            } else if (num > small) {
                small = num;
            }
        }
        return count < 3 ? max : small;//has three different number
    }

    //priority queue and set
    public int thirdMaxQueue(int[] nums) {
        PriorityQueue<Integer> pq = new PriorityQueue<Integer>();
        Set<Integer> set = new HashSet<Integer>();
        for (int x : nums) {
            if (!set.contains(x)) {
                pq.offer(x);
                set.add(x);
                if (pq.size() > 3) {
                    set.remove(pq.poll());
                }
            }
        }
        if (pq.size() < 3) {
            while (pq.size() > 1) {
                pq.poll();
            }
        }
        return pq.peek();
    }


    //quick select
    //find the kth minimum number in an array
    //first we should write the quick sort algo
    public void quickSort(int[] nums, int left, int right) {
        if (left < right) {
            //快速排序的最坏情况基于每次划分对主元的选择。基本的快速排序选取第一个元素作为主元。这样在数组已经有序的情况下，每次划分将得到最坏的结果。一种比较常见的优化方法是随机化算法
            //Random rand=new Random();
            //int index=rand.nextInt(right-left+1)+left;
            //int begin=left,end=right,key=nums[index];
            //swap nums[begin],nums[index];
            //int tmp=nums[index];
            // nums[index]=nums[begin];
            // nums[begin]=tmp;
            int begin = left, end = right, key = nums[begin];
            while (begin < end) {
                while (begin < end && nums[end] >= key)
                    end--;
                nums[begin] = nums[end];
                while (begin < end && nums[begin] <= key)//change operator to sort reverse or not
                    begin++;
                nums[end] = nums[begin];
            }
            nums[begin] = key;
            quickSort(nums, left, begin - 1);
            quickSort(nums, begin + 1, right);
        }
    }

    public void quickSort(int[] nums) {
        if (nums == null || nums.length <= 1)
            return;
        quickSort(nums, 0, nums.length - 1);
        for (int x : nums) {
            System.out.println(x);
        }
    }


    //66 plus one
    public int[] plusOne(int[] digits) {
        int n = digits.length;
        for (int i = n - 1; i >= 0; --i) {
            if (digits[i] != 9) {
                digits[i]++;
                return digits;
            } else
                digits[i] = 0;
        }
        int[] res = new int[n + 1];//default value is 0;
        res[0] = 1;
        return res;
    }

    //you should think linkedlist, it just traverse from the end, you can definitely reverse and do something
    //but the better way is do something

    //217 contains duplicate
    //O(N) and O(N)
    public boolean containsDuplicate(int[] nums) {
        Map<Integer, Integer> map = new HashMap<Integer, Integer>();
        for (int x : nums) {
            if (!map.containsKey(x)) {
                map.put(x, 1);
            } else {
                map.put(x, map.get(x) + 1);
                if (map.get(x) > 1)
                    return true;
            }
        }
        return false;
    }

    //hashset is more concise
    public boolean containsDuplicates(int[] nums) {
        Set<Integer> set = new HashSet<Integer>();
        for (int x : nums) {
            if (!set.add(x))
                return true;
        }
        return false;
    }
    //you can also sort first and check whether there is a duplicae
    //O(nlogn) and O(1)
    //return nums.size() > set<int>(nums.begin(), nums.end()).size();
    //hahaha


    //219 contains duplicate ii
    public boolean containsNearbyDuplicate(int[] nums, int k) {
        Map<Integer, Integer> map = new HashMap<Integer, Integer>();
        int n = nums.length;
        for (int i = 0; i < n; ++i) {
            if (map.containsKey(nums[i])) {
                if (i - map.get(nums[i]) <= k)
                    return true;
            }
            map.put(nums[i], i);
        }
        return false;
    }

    //the better way is use set, remember k you should always use window
    public boolean containsNearbyDuplicateSet(int[] nums, int k) {
        Set<Integer> set = new HashSet<Integer>();
        int n = nums.length;
        for (int i = 0; i < n; ++i) {
            if (i > k)
                set.remove(nums[i - k - 1]);//think for a while
            if (!set.add(nums[i]))
                return true;
        }
        return false;
    }

    //220 contains duplicate III
    public boolean containsNearByAlmostDuplicate(int[] nums, int k, int t) {
        if (nums.length == 0 || k < 0)
            return false;
        final TreeSet<Long> values = new TreeSet<>();
        int n = nums.length;
        for (int ind = 0; ind < n; ++ind) {
            final Long floor = values.floor((long) nums[ind] + t);
            final Long ceil = values.ceiling((long) nums[ind] - t);
            if (floor != null && floor >= nums[ind] || (ceil != null && ceil <= nums[ind]))
                return true;
            values.add((long) nums[ind]);
            if (ind >= k)//here you can change ind>k and remove nums[ind-k-1] and move this part code to upper
                values.remove(nums[ind - k]);
        }
        return false;
    }



    //88 merge sorted array
    public void merge(int[] nums1, int m, int[] nums2, int n) {
        int k = m + n - 1;
        int i = m - 1;
        int j = n - 1;
        while (i >= 0 && j >= 0) {
            if (nums1[i] >= nums2[j])
                nums1[k--] = nums1[i--];
            else
                nums1[k--] = nums2[j--];
        }
        while (j >= 0) {
            nums1[k--] = nums2[j--];
        }

    }

    //more concise
    public void mergeConcise(int[] nums1, int m, int[] nums2, int n) {
        int k = m + n - 1;
        int i = m - 1;
        int j = n - 1;
        while (j >= 0) {
            nums1[k--] = i >= 0 && nums1[i] >= nums2[j] ? nums1[i--] : nums2[j--];
        }
    }

    //26 remove duplicates from sorted array
    public int removeDuplicates(int[] nums) {
        int n = nums.length;
        if (n <= 1)
            return n;
        int i = 1, j = 1;
        while (i < n) {
            if (nums[i] != nums[i - 1])
                nums[j++] = nums[i];
            i++;
        }
        return j;
    }

    //calculate the count
    public int removeDuplicateByCount(int[] nums) {
        int count = 0, n = nums.length;
        for (int i = 1; i < n; ++i) {
            if (nums[i] == nums[i - 1])
                count++;
            else
                nums[i - count] = nums[i];
        }
        return n - count;
    }

    //27 remove element
    public int removeElement(int[] nums, int val) {
        int n = nums.length;
        int i = 0, j = 0;
        while (i < n) {
            if (nums[i] != val) {
                nums[j++] = nums[i];
            }
            i++;
        }
        return j;
    }

    //rotata array
    //at least 3 ways
    //google guile gcd
    public void reverse(int[] nums, int l, int r) {
        while (l < r) {
            int tmp = nums[l];
            nums[l++] = nums[r];
            nums[r--] = tmp;
        }
    }

    public void rotate(int[] nums, int k) {
        int n = nums.length;
        if (k % n == 0)
            return;
        k = n - k % n;
        reverse(nums, 0, k - 1);
        reverse(nums, k, n - 1);
        reverse(nums, 0, n - 1);
    }


    //anther way, o(n) space
    public void rotate2(int[] nums, int k) {
        int n = nums.length;
        if (k % n == 0)
            return;
        k = k % n;
        int copy[] = new int[n];
        for (int i = 0; i < n; ++i) {
            copy[i] = nums[i];
        }
        for (int i = 0; i < n; i++) {
            nums[(i + k) % n] = copy[i];
        }
    }

    //the most efficient way

    public int gcd(int a, int b) {
        return b == 0 ? a : gcd(b, a % b);
    }

    public void leftRotate(int[] nums, int d) {
        int n = nums.length, tmp = 0, k = 0;
        int gcdNumber = gcd(n, d);
        for (int i = 0; i < gcdNumber; ++i) {
            tmp = nums[i];
            int j = i;
            while (true) {
                k = (j + d) % n;
                if (k == i)
                    break;
                nums[j] = nums[k];
                j = k;
            }
            nums[j] = tmp;
        }

    }
    //as for the right rotate, you can use leftrotate,or just write the similar code



    //118 pascal's triangle
    public List<List<Integer>> generate(int numRows) {
        List<List<Integer>> res = new ArrayList<List<Integer>>();
        if (numRows <= 0)
            return res;
        List<Integer> path = new ArrayList<Integer>();
        path.add(1);
        res.add(path);
        for (int i = 1; i < numRows; ++i) {
            List<Integer> newRow = new ArrayList<Integer>();
            newRow.add(1);
            for (int j = 1; j < res.get(i - 1).size(); ++j)
                newRow.add(res.get(i - 1).get(j - 1) + res.get(i - 1).get(j));
            newRow.add(1);
            res.add(newRow);
        }
        return res;
    }

    //more concise
    public List<List<Integer>> generateConcise(int numRows) {
        List<List<Integer>> triangle = new ArrayList<List<Integer>>();
        if (numRows <= 0) {
            return triangle;
        }
        for (int i = 0; i < numRows; i++) {
            List<Integer> row = new ArrayList<Integer>();
            for (int j = 0; j < i + 1; j++) {
                if (j == 0 || j == i) {
                    row.add(1);
                } else {
                    row.add(triangle.get(i - 1).get(j - 1) + triangle.get(i - 1).get(j));
                }
            }
            triangle.add(row);
        }
        return triangle;
    }

    //pascal's triangel II
    public List<Integer> getRow(int rowIndex) {
        List<Integer> res = new ArrayList<Integer>();
        for (int i = 0; i <= rowIndex; ++i) {
            res.add(0, 1);
            for (int j = 1; j < res.size() - 1; ++j)
                res.set(j, res.get(j) + res.get(j + 1));
        }
        return res;
    }

    //moer space
    public List<Integer> getRowMoreSpace(int rowIndex) {
        List<Integer> res = new ArrayList<Integer>();
        res.add(1);
        for (int i = 1; i <= rowIndex; ++i) {
            List<Integer> newRow = new ArrayList<Integer>();
            newRow.add(1);
            for (int j = 1; j < res.size(); ++j)
                newRow.add(res.get(j - 1) + res.get(j));
            newRow.add(1);
            res = newRow;
        }
        return res;
    }

    //448  Find All Numbers Disappeared in an Array
    //but not O(1) space
    //442
    public List<Integer> findDisappearedNumbers(int[] nums) {
        List<Integer> res = new ArrayList<Integer>();
        Set<Integer> set = new HashSet<Integer>();
        int n = nums.length;
        for (int x : nums)
            set.add(x);
        for (int i = 1; i <= n; ++i) {
            if (!set.contains(i))
                res.add(i);
        }
        return res;
    }

    //interesting question
    public List<Integer> findDisappearNumbersConcise(int[] nums) {
        //mark these number who have existed in the array
        List<Integer> ret = new ArrayList<Integer>();
        for (int i = 0; i < nums.length; i++) {
            int val = Math.abs(nums[i]) - 1;
            if (nums[val] > 0) {
                nums[val] = -nums[val];
            }
        }

        for (int i = 0; i < nums.length; i++) {
            if (nums[i] > 0) {
                ret.add(i + 1);
            }
        }
        return ret;
    }

    //medium

    //152 maximum product array, I am not in status today
    public int maxProduct(int[] nums) {
        if (nums == null || nums.length == 0)
            return Integer.MIN_VALUE;
        int n = nums.length;
        int[] maximum = new int[n];
        int[] minimum = new int[n];
        maximum[0] = minimum[0] = nums[0];
        int res = nums[0];
        for (int i = 1; i < n; ++i) {
            maximum[i] = Math.max(nums[i], Math.max(maximum[i - 1] * nums[i], minimum[i - 1] * nums[i]));
            minimum[i] = Math.min(nums[i], Math.min(maximum[i - 1] * nums[i], minimum[i - 1] * nums[i]));
            res = Math.max(res, maximum[i]);
        }
        return res;
    }

    //save space
    public int maxPorductSaveSpace(int[] nums) {
        if (nums == null || nums.length == 0)
            return Integer.MIN_VALUE;
        int n = nums.length;
        int maxi = nums[0], mini = nums[0], saveMaxi = 0;
        int res = nums[0];
        for (int i = 1; i < n; ++i) {
            saveMaxi = maxi;
            maxi = Math.max(nums[i], Math.max(maxi * nums[i], mini * nums[i]));
            mini = Math.min(nums[i], Math.min(saveMaxi * nums[i], mini * nums[i]));
            res = Math.max(res, maxi);
        }
        return res;
    }

    //more concise
    public void swap(int[] nums, int i, int j) {
        int tmp = nums[i];
        nums[i] = nums[j];
        nums[j] = tmp;
    }

    public int maxProductConcise(int[] nums) {
        int r = nums[0], imax = r, imin = r;
        int n = nums.length;
        int res = nums[0];
        for (int i = 1; i < n; ++i) {
            if (nums[i] < 0) {
                int tmp = imax;
                imax = imin;
                imin = tmp;
            }
            imax = Math.max(nums[i], imax * nums[i]);
            imin = Math.min(nums[i], imin * nums[i]);
            res = Math.max(res, imax);
        }
        return res;
    }

    //228 summary ranges
    //interesting
    public List<String> summaryRanges(int[] nums) {
        List<String> res = new ArrayList<String>();
        if (nums == null || nums.length == 0)
            return res;
        int n = nums.length;
        int start = nums[0];
        for (int i = 0; i < n; ++i) {
            if (i == n - 1 || nums[i + 1] != nums[i] + 1) {//this can used to remove duplicate code
                if (nums[i] == start)
                    res.add(String.valueOf(start));
                else
                    res.add(start + "->" + nums[i]);
                if (i != n - 1)
                    start = nums[i + 1];
            }
        }
        return res;
    }

    //216 && 39 && 40
    // Combination Sum serials
    public void backtrack(List<List<Integer>> res, List<Integer> path, int target, int index, int[] candidates) {
        if (target == 0) {
            res.add(new ArrayList<Integer>(path));
            return;
        }
        for (int i = index; i < candidates.length; ++i) {
            if (target >= candidates[i]) {
                path.add(candidates[i]);
                backtrack(res, path, target - candidates[i], i, candidates);
                path.remove(path.size() - 1);
            }
        }
    }
    public List<List<Integer>> combinationSum(int[] candidates, int target) {
        List<List<Integer>> res = new ArrayList<List<Integer>>();
        List<Integer> path = new ArrayList<Integer>();
        backtrack(res, path, target, 0, candidates);
        return res;
    }

    //40 combination sum ii

    public void backtrack2(List<List<Integer>> res, List<Integer> path, int target, int index, int[] candidates) {
        if (target == 0) {
            res.add(new ArrayList<Integer>(path));
            return;
        }
        for (int i = index; i < candidates.length; ++i) {

            if (i != index && candidates[i] == candidates[i - 1])//skip duplicates
                continue;
            if (target >= candidates[i]) {
                path.add(candidates[i]);
                backtrack2(res, path, target - candidates[i], i + 1, candidates);
                path.remove(path.size() - 1);
            } else
                break;
        }
    }

    public List<List<Integer>> combinationSum2(int[] candidates, int target) {
        List<List<Integer>> res = new ArrayList<List<Integer>>();
        List<Integer> path = new ArrayList<Integer>();
        Arrays.sort(candidates);
        backtrack2(res, path, target, 0, candidates);
        return res;
    }


    //216 combination sum III
    public void backtrack3(List<List<Integer>> res, List<Integer> path, int target, int index, int k) {
        if (target == 0 && path.size() == k) {
            res.add(new ArrayList<Integer>(path));
            return;
        }
        for (int i = index; i <= 9; ++i) {
            if (target >= i) {
                path.add(i);
                backtrack3(res, path, target - i, i + 1, k);
                path.remove(path.size() - 1);
            } else
                break;
        }
    }

    public List<List<Integer>> combinationSum3(int k, int n) {
        List<List<Integer>> res = new ArrayList<List<Integer>>();
        List<Integer> path = new ArrayList<Integer>();
        backtrack3(res, path, n, 1, k);
        return res;
    }


    //377 combinationSum4

    public int combinationSum4(int[] nums, int target) {
        int n = nums.length;
        int[] dp = new int[target + 1];
        Arrays.sort(nums);
        dp[0] = 1;
        for (int i = 1; i <= target; ++i) {
            for (int j = 0; j < n; ++j) {
                if (i >= nums[j]) {
                    dp[i] += dp[i - nums[j]];
                } else
                    break;
            }
        }
        return dp[target];
    }

    //48 rotate image
    //clockwise
    public void rotate(int[][] matrix) {
        if (matrix.length == 0 || matrix[0].length == 0)
            return;
        //reverse the array
        int m = matrix.length, n = matrix[0].length;
        int begin = 0, end = m - 1;
        while (begin < end) {
            int[] tmp = matrix[begin];
            matrix[begin++] = matrix[end];
            matrix[end--] = tmp;
        }

        //reverse the diagnoal
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < i; ++j) {
                int tmp = matrix[i][j];
                matrix[i][j] = matrix[j][i];
                matrix[j][i] = tmp;
            }
        }
    }

    public void swap(int[] nums) {
        if (nums == null || nums.length <= 1)
            return;
        int begin = 0, end = nums.length - 1;
        while (begin < end) {
            int tmp = nums[begin];
            nums[begin++] = nums[end];
            nums[end--] = tmp;
        }
    }

    //follow up rotate counterclockwise
    public void rotateCounterclockWise(int[][] matrix) {
        if (matrix.length == 0 || matrix[0].length == 0)
            return;
        //reverse the array
        int m = matrix.length, n = matrix[0].length;
        for (int i = 0; i < m; ++i)
            swap(matrix[i]);
        //reverse the diagnoal
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < i; ++j) {
                int tmp = matrix[i][j];
                matrix[i][j] = matrix[j][i];
                matrix[j][i] = tmp;
            }
        }
    }


    //container with most water
    public int maxArea(int[] height) {
        int maxArea = 0;
        int begin = 0, end = height.length - 1;
        while (begin < end) {
            if (height[begin] < height[end]) {
                maxArea = Math.max(maxArea, height[begin] * (end - begin));
                begin++;
            } else {
                maxArea = Math.max(maxArea, height[end] * (end - begin));
                end--;
            }
        }
        return maxArea;
    }

    //we can move further
    public int maxAreaBetter(int[] height) {
        int maxArea = 0;
        int begin = 0, end = height.length - 1;
        while (begin < end) {
            int h = Math.min(height[begin], height[end]);
            maxArea = Math.max(maxArea, (end - begin) * h);
            while (begin < end && height[begin] <= h)
                begin++;
            while (begin < end && height[end] <= h)
                end--;
        }
        return maxArea;
    }

    //209 minimum size subarray sum
    //two window
    public int minSubArrayLen(int s, int[] nums) {
        if (nums == null || nums.length == 0)
            return 0;
        int res = Integer.MAX_VALUE, sum = 0, n = nums.length;
        int begin = 0, end = 0;
        while (end < n) {
            sum += nums[end++];
            while (sum >= s) {
                if (res > end - begin) {//do remember that end is out of boundary, so just end -begin is ok
                    res = end - begin;
                }
                sum -= nums[begin++];
            }
        }
        return res == Integer.MAX_VALUE ? 0 : res;
    }

    //you could also get the sum array and binary search


    //238 Product of Array Except Self
    public int[] productExceptSelf(int[] nums) {
        if (nums == null || nums.length == 0)
            return nums;
        int n = nums.length;
        int[] res = new int[n];
        res[0] = 1;
        for (int i = 1; i < n; ++i)
            res[i] = res[i - 1] * nums[i - 1];
        int right = 1;
        for (int i = n - 1; i >= 0; --i) {
            res[i] *= right;
            right *= nums[i];
        }
        return res;

    }

    //268 missing number
    //bit manipulation
    //sum up
    //binary search
    //If the array is in order, I prefer Binary Search method. Otherwise, the XOR method is better.
    public int missingNumber(int[] nums) {
        int res = 0;
        int n = nums.length;
        for (int i = 1; i <= n; ++i) {
            res ^= nums[i - 1];
            res ^= i;
        }
        return res;
    }

    //binary search
    public int missingNumberBinarySearch(int[] nums) {
        Arrays.sort(nums);
        int begin = 0, end = nums.length;
        while (begin < end) {
            int mid = (end - begin) / 2 + begin;
            if (nums[mid] > mid)
                end = mid;
            else
                begin = mid + 1;
        }
        return begin;
    }

    //you can sum them up
    public int missingNumberUsingSumUp(int[] nums) {
        int len = nums.length;
        int sum = (0 + len) * (len + 1) / 2;
        for (int i = 0; i < len; ++i)
            sum -= nums[i];
        return sum;
    }

    //in case of overflow
    public int missingNumberOverflow(int[] nums) {
        int sum = 0;
        for (int i = 0; i < nums.length; ++i) {
            sum += nums[i] - i;
        }
        return nums.length - sum;
    }

    //259 3sum smaller
    public int threeSumSmaller(int[] nums, int target) {
        if (nums == null || nums.length == 0)
            return 0;
        Arrays.sort(nums);
        int n = nums.length;
        int res = 0;
        for (int i = 0; i < n - 2; ++i) {
            int begin = i + 1, end = n - 1;
            while (begin < end) {
                while (begin < end && nums[i] + nums[begin] + nums[end] >= target)
                    end--;
                res += end - begin;
                //end=n-1;//don't need to find change end to n-1 again, because begin++,so end must --
                begin++;
            }
        }
        return res;
    }

    //without inner loop
    public int threeSumSmallerWithoutInnerLoop(int[] nums, int target) {
        int n = nums.length;
        Arrays.sort(nums);
        int count = 0;
        for (int i = 0; i < n - 2; ++i) {
            int lo = i + 1;
            int hi = n - 1;
            while (lo < hi) {
                if (nums[i] + nums[lo] + nums[hi] < target) {
                    count += hi - lo;
                    lo++;
                } else
                    hi--;
            }
        }
        return count;
    }

    //54 spiral matrix
    //interesting question
    //clockwise
    //O(n) space
    public List<Integer> spiralOrder(int[][] matrix) {
        List<Integer> res = new ArrayList<Integer>();
        if (matrix == null || matrix.length == 0 || matrix[0].length == 0)
            return res;
        int m = matrix.length, n = matrix[0].length;
        int row = 0, col = 0;
        boolean[][] vis = new boolean[m][n];
        res.add(matrix[row][col]);
        vis[row][col] = true;
        while (res.size() < m * n) {
            while (col + 1 < n && !vis[row][col + 1]) {
                res.add(matrix[row][++col]);
                vis[row][col] = true;
            }
            while (row + 1 < m && !vis[row + 1][col]) {
                res.add(matrix[++row][col]);
                vis[row][col] = true;
            }

            while (col > 0 && !vis[row][col - 1]) {
                res.add(matrix[row][--col]);
                vis[row][col] = true;
            }

            while (row > 0 && !vis[row - 1][col]) {
                res.add(matrix[--row][col]);
                vis[row][col] = true;
            }
        }
        return res;
    }

    //if it is counterclockwise?
    public List<Integer> spiralOrderCounterClockwise(int[][] matrix) {
        List<Integer> res = new ArrayList<Integer>();
        if (matrix == null || matrix.length == 0 || matrix[0].length == 0)
            return res;
        int m = matrix.length, n = matrix[0].length;
        int row = 0, col = 0;
        boolean[][] vis = new boolean[m][n];
        res.add(matrix[row][col]);
        vis[row][col] = true;
        while (res.size() < m * n) {
            while (row + 1 < m && !vis[row + 1][col]) {
                res.add(matrix[++row][col]);
                vis[row][col] = true;
            }
            while (col + 1 < n && !vis[row][col + 1]) {
                res.add(matrix[row][++col]);
                vis[row][col] = true;
            }
            while (row > 0 && !vis[row - 1][col]) {
                res.add(matrix[--row][col]);
                vis[row][col] = true;
            }
            while (col > 0 && !vis[row][col - 1]) {
                res.add(matrix[row][--col]);
                vis[row][col] = true;
            }
        }
        return res;
    }


    //O(1) space
    public List<Integer> spiralOrderSaveSpace(int[][] matrix) {
        List<Integer> res = new ArrayList<Integer>();

        if (matrix.length == 0) {
            return res;
        }

        int rowBegin = 0;
        int rowEnd = matrix.length - 1;
        int colBegin = 0;
        int colEnd = matrix[0].length - 1;

        while (rowBegin <= rowEnd && colBegin <= colEnd) {
            // Traverse Right
            for (int j = colBegin; j <= colEnd; j++) {
                res.add(matrix[rowBegin][j]);
            }
            rowBegin++;

            // Traverse Down
            for (int j = rowBegin; j <= rowEnd; j++) {
                res.add(matrix[j][colEnd]);
            }
            colEnd--;

            if (rowBegin <= rowEnd) {//the condition is rowBegin<=rowEnd, so after rowBegin++, it is possible destory the condition.
                // Traverse Left
                for (int j = colEnd; j >= colBegin; j--) {
                    res.add(matrix[rowEnd][j]);
                }
            }
            rowEnd--;

            if (colBegin <= colEnd) {
                // Traver Up
                for (int j = rowEnd; j >= rowBegin; j--) {
                    res.add(matrix[j][colBegin]);
                }
            }
            colBegin++;
        }
        return res;
    }

    //59 spiral matrix II
    public int[][] generateMatrix(int n) {
        int[][] res = new int[n][n];
        int count = 1;
        int rowBeg = 0, rowEnd = n - 1, colBeg = 0, colEnd = n - 1;
        while (rowBeg <= rowEnd && colBeg <= colEnd) {
            //traverse the left
            for (int i = colBeg; i <= colEnd; ++i)
                res[rowBeg][i] = count++;
            rowBeg++;

            //down
            for (int i = rowBeg; i <= rowEnd; ++i)
                res[i][colEnd] = count++;
            colEnd--;

            if (rowBeg <= rowEnd) {
                for (int i = colEnd; i >= colBeg; --i)
                    res[rowEnd][i] = count++;
            }
            rowEnd--;

            if (colBeg <= colEnd) {
                for (int i = rowEnd; i >= rowBeg; --i)
                    res[i][colBeg] = count++;
            }
            colBeg++;
        }
        return res;
    }

    //79 word search
    //O(N*N) space
    //won't change board

    int[] dx = {1, -1, 0, 0};
    int[] dy = {0, 0, 1, -1};

    public boolean dfsExist(char[][] board, String word, int pos, boolean[][] vis, int x, int y) {
        if (pos == word.length())
            return true;
        for (int k = 0; k < 4; ++k) {
            int xx = x + dx[k];
            int yy = y + dy[k];
            if (xx < 0 || xx >= board.length || yy < 0 || yy >= board[0].length || vis[xx][yy] || word.charAt(pos) != board[xx][yy])
                continue;
            vis[xx][yy] = true;
            if (dfsExist(board, word, pos + 1, vis, xx, yy))
                return true;
            vis[xx][yy] = false;
        }
        return false;
    }

    public boolean exist(char[][] board, String word) {
        if (board == null || board.length == 0 || board[0].length == 0 || word.length() == 0)
            return false;
        int m = board.length, n = board[0].length;
        boolean[][] vis = new boolean[m][n];
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (!vis[i][j] && board[i][j] == word.charAt(0)) {
                    vis[i][j] = true;
                    if (dfsExist(board, word, 1, vis, i, j))
                        return true;
                    vis[i][j] = false;

                }
            }
        }
        return false;
    }

    //change board to save space
    //Your code works, but when your algorithm is running, board[][] is changed, which prevents it from concurrent usage.

    //
//    private boolean exist(char[][] board, int y, int x, char[] word, int i) {
//        if (i == word.length) return true;
//        if (y<0 || x<0 || y == board.length || x == board[y].length) return false;
//        if (board[y][x] != word[i]) return false;
//        board[y][x] ^= 256;
//        boolean exist = exist(board, y, x+1, word, i+1)
//                || exist(board, y, x-1, word, i+1)
//                || exist(board, y+1, x, word, i+1)
//                || exist(board, y-1, x, word, i+1);
//        board[y][x] ^= 256;
//        return exist;
//    }
    public boolean exist(char[][] board, int i, int j, String word, int pos) {
        if (pos == word.length())
            return true;
        if (i < 0 || i >= board.length || j < 0 || j >= board[0].length || board[i][j] != word.charAt(pos))
            return false;
        board[i][j] = '*';
        boolean result = exist(board, i - 1, j, word, pos + 1) || exist(board, i + 1, j, word, pos + 1) || exist(board, i, j + 1, word, pos + 1) || exist(board, i, j - 1, word, pos + 1);
        board[i][j] = word.charAt(pos);
        return result;
    }

    public boolean existSaveSpace(char[][] board, String word) {
        if (board == null || board.length == 0 || board[0].length == 0 || word.length() == 0)
            return false;
        int m = board.length, n = board[0].length;
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (exist(board, i, j, word, 0))
                    return true;
            }
        }
        return false;
    }

    //120 triangle
    //change the value of the triangle
    public int minimumTotal(List<List<Integer>> triangle) {
        if (triangle == null || triangle.isEmpty() || triangle.get(0).isEmpty())
            return Integer.MAX_VALUE;
        int m = triangle.size();
        for (int i = m - 2; i >= 0; i--) {
            for (int j = 0; j < triangle.get(i).size(); ++j) {
                int val = Math.min(triangle.get(i + 1).get(j), triangle.get(i + 1).get(j + 1));
                triangle.get(i).set(j, triangle.get(i).get(j) + val);
            }
        }
        return triangle.get(0).get(0);
    }

    //without changing triangle
    public int minimumTotalWithoutChange(List<List<Integer>> triangle) {
        int n = triangle.size();
        List<Integer> minlen = new ArrayList<Integer>(triangle.get(n - 1));//copy the last level
        for (int layer = n - 2; layer >= 0; --layer) {
            for (int i = 0; i <= layer; ++i) {
                minlen.set(i, Math.min(minlen.get(i), minlen.get(i + 1)) + triangle.get(layer).get(i));
            }
        }
        return minlen.get(0);
    }

    //162 find peak element
    //If num[i-1] < num[i] > num[i+1], then num[i] is peak
    //If num[i-1] < num[i] < num[i+1], then num[i+1...n-1] must contains a peak
    //If num[i-1] > num[i] > num[i+1], then num[0...i-1] must contains a peak
    //If num[i-1] > num[i] < num[i+1], then both sides have peak
    //num[i] ≠ num[i+1]
    //interseting question
    public int findPeakElement(int[] nums) {
        int begin = 0, end = nums.length - 1;
        while (begin < end) {
            int mid = (end - begin) / 2 + begin;//mid maxvalue is end-1;
            if (nums[mid + 1] > nums[mid])//so mid+1, not mid and mid-1
                begin = mid + 1;
            else
                end = mid;
        }
        return begin;
    }

    //442 find all duplicates in array
    //hashmap
    public List<Integer> findDuplicates(int[] nums) {
        //Map<Integer,Integer>map=new HashMap<Integer, Integer>();
        Set<Integer> set = new HashSet<Integer>();
        List<Integer> res = new ArrayList<Integer>();
        for (int x : nums) {
            if (!set.add(x))
                res.add(x);
        }
        return res;
    }

    //also you can sort
    //you can use O(N) to sort
    public List<Integer> findDuplicatesBysort(int[] nums) {
        Arrays.sort(nums);
        int n = nums.length;
        List<Integer> res = new ArrayList<Integer>();
        for (int i = 1; i < n; ++i) {
            if (nums[i] == nums[i - 1]) {
                res.add(nums[i]);
            }
        }
        return res;
    }

    //the same as find disapperaing number leetcode 448
    public List<Integer> findDuplicatesSimple(int[] nums) {
        List<Integer> res = new ArrayList<Integer>();
        int n = nums.length;
        for (int i = 0; i < n; ++i) {
            int val = Math.abs(nums[i]) - 1;
            if (nums[val] < 0)
                res.add(Math.abs(val + 1));
            nums[val] = -nums[val];
        }
        return res;
    }

    //75 sort color
    //荷兰国旗
    public void sortColors(int[] nums) {
        int begin = 0, cur = 0, end = nums.length - 1;
        while (cur <= end) {
            if (nums[cur] == 0) {
                swap(nums, cur, begin);
                cur++;
                begin++;
            } else if (nums[cur] == 2) {
                swap(nums, cur, end);
                end--;
            } else
                cur++;
        }
    }

    //122 best time to buy and sell stokcii
    public int maxProfitII(int[] prices) {
        int n = prices.length;
        int maxProfit = 0;
        for (int i = 1; i < n; ++i)
            if (prices[i] > prices[i - 1])
                maxProfit += prices[i] - prices[i - 1];
        return maxProfit;
    }

    //153 find minimum in rotated sorted array
    public int findMin(int[] nums) {
        int n = nums.length;
        int begin = 0, end = n - 1;
        while (begin < end) {
            if (nums[begin] < nums[end])
                return nums[begin];
            int mid = (end - begin) / 2 + begin;
            if (nums[mid] > nums[end])//really good question,mid can equal begin, but can only reach end-1; so >begin is wrong
                begin = mid + 1;
            else
                end = mid;
        }
        return nums[begin];
    }

    //154 find minimum in rotated array II
    //
    public int findMinII(int[] nums) {
        int n = nums.length;
        int begin = 0, end = n - 1;
        while (begin < end) {
            if (nums[begin] < nums[end])
                return nums[begin];
            int mid = (end - begin) / 2 + begin;
            if (nums[mid] > nums[end])//really goode question,mid can equal begin, but can only reach end-1;
                begin = mid + 1;
            else if (nums[mid] < nums[end])
                end = mid;
            else
                end--;//When num[mid] == num[hi], we couldn't sure the position of minimum in mid's left or right, so just let upper bound reduce one.
        }
        return nums[begin];
    }

    //62 Unique Paths, actually  it is a combination number

    public int combination(int n, int k) {
        if (k < n - k)//reduce the multiple times
            k = n - k;
        long res = 1;
        for (int i = n - k + 1; i <= n; ++i)
            res *= i;
        for (int i = 1; i <= k; ++i)
            res /= i;
        return (int) res;
    }

    public int uniquePaths(int m, int n) {
        if (m < 1 || m + n < 2 || n < 1)
            return 1;
        return combination(m + n - 2, m - 1);

    }

    //dynamic programming
    //may be you can find some way to sace some space
    public int uniquePathsDp(int m, int n) {
        if (m < 1 || m + n < 2 || n < 1)
            return 1;
        int[][] dp = new int[m][n];
        dp[0][0] = 1;//in case of there is only one position 0,0;
        for (int i = m - 2; i >= 0; --i)
            dp[i][n - 1] = 1;
        for (int i = n - 2; i >= 0; --i)
            dp[m - 1][i] = 1;
        for (int i = m - 2; i >= 0; --i) {
            for (int j = n - 2; j >= 0; --j)
                dp[i][j] = dp[i + 1][j] + dp[i][j + 1];
        }
        return dp[0][0];
    }

    //63 Follow up for "Unique Paths":
    //unique pathII
    //may be you can think some way to save space
    public int uniquePathsWithObstacles(int[][] obstacleGrid) {
        if (obstacleGrid == null || obstacleGrid.length == 0 || obstacleGrid[0].length == 0)
            return 0;
        int m = obstacleGrid.length, n = obstacleGrid[0].length;
        if (obstacleGrid[0][0] == 1 || obstacleGrid[m - 1][n - 1] == 1)
            return 0;
        int[][] dp = new int[m][n];
        dp[m - 1][n - 1] = 1;
        for (int i = m - 2; i >= 0; --i) {
            dp[i][n - 1] = obstacleGrid[i][n - 1] == 1 ? 0 : dp[i + 1][n - 1];
        }

        for (int i = n - 2; i >= 0; --i) {
            dp[m - 1][i] = obstacleGrid[m - 1][i] == 1 ? 0 : dp[m - 1][i + 1];
        }

        for (int i = m - 2; i >= 0; --i) {
            for (int j = n - 2; j >= 0; --j)
                dp[i][j] = obstacleGrid[i][j] == 1 ? 0 : dp[i + 1][j] + dp[i][j + 1];
        }
        return dp[0][0];
    }


    //64 minimum path sum
    public int minPathSum(int[][] grid) {
        if (grid == null || grid.length == 0 || grid[0].length == 0)
            return 0;
        int m = grid.length, n = grid[0].length;
        int[][] dp = new int[m][n];
        dp[m - 1][n - 1] = grid[m - 1][n - 1];
        for (int i = m - 2; i >= 0; --i) {
            dp[i][n - 1] = grid[i][n - 1] + dp[i + 1][n - 1];
        }

        for (int i = n - 2; i >= 0; --i) {
            dp[m - 1][i] = grid[m - 1][i] + dp[m - 1][i + 1];
        }

        for (int i = m - 2; i >= 0; --i) {
            for (int j = n - 2; j >= 0; --j)
                dp[i][j] = grid[i][j] + Math.min(dp[i + 1][j], dp[i][j + 1]);
        }
        return dp[0][0];
    }


    //35 search insert position
    //lower_bound
    public int searchInsertPosition(int[] nums, int target) {
        int n = nums.length;
        if (nums[n - 1] < target)
            return n;
        int begin = 0, index = 0, end = n - 1;
        while (begin < end) {
            int mid = (end - begin) / 2 + begin;
            if (nums[mid] < target) {
                begin = mid + 1;
                index = begin;
            } else {
                end = mid;
                index = end;
            }
        }
        return index;
    }

    //find the celebrity
    public boolean knows(int a, int b) {
        return true;
    }

    public int findCelebrity(int n) {
        int c = 0;
        for (int i = 1; i < n; ++i) {
            if (knows(c, i))
                c = i;
        }
        //check whether c is a valid
        for (int i = 0; i < n; ++i) {
            if (i != c) {
                if (knows(c, i) || !knows(i, c))
                    return -1;
            }
        }
        return c;
    }

    //74 search in a 2D matrix
    //very interesting, you should always check the last line;
    // it is /n not /m , you should think it about
    public boolean searchMatrix(int[][] matrix, int target) {
        if (matrix == null || matrix.length == 0 || matrix[0].length == 0)
            return false;
        int m = matrix.length, n = matrix[0].length;
        int begin = 0, end = m * n - 1;
        while (begin < end) {
            int mid = (end - begin) / 2 + begin;
            if (matrix[mid / n][mid % n] == target)
                return true;
            else if (matrix[mid / n][mid % n] > target)
                end = mid;
            else
                begin = mid + 1;
        }
        return matrix[begin / n][begin % n] == target;//in my kind of binary search ,I need to check before return
    }

    //240 search a 2d matrix II
    //this question is interesting
    //Hence worst case scenario is O(m+n).
    public boolean searchMatrixII(int[][] matrix, int target) {
        if (matrix == null || matrix.length == 0 || matrix[0].length == 0)
            return false;
        int m = matrix.length, n = matrix[0].length;
        int row = 0, col = n - 1;
        while (col >= 0 && row < m) {
            if (matrix[row][col] == target)
                return true;
            else if (matrix[row][col] > target)
                col--;
            else
                row++;
        }
        return false;

    }

    //280 wiggle sort
    //interesting question
    //odd and even property
    public void wiggleSort(int[] nums) {
        int n = nums.length;
        for (int i = 1; i < n; ++i) {
            if (i % 2 == 1 && nums[i] < nums[i - 1]) {
                swap(nums, i - 1, i);
            } else if (i % 2 == 0 && nums[i - 1] < nums[i]) {
                swap(nums, i - 1, i);
            }
        }
    }


    //78 subsets
    //given a set of integer, nums,return all possible subsets

    //backtracking
    public void backtrackSubset(List<List<Integer>> res, List<Integer> path, int[] nums, int pos) {
        res.add(new ArrayList<Integer>(path));
        for (int i = pos; i < nums.length; ++i) {
            path.add(nums[i]);
            backtrackSubset(res, path, nums, i + 1);
            path.remove(path.size() - 1);
        }
    }

    public List<List<Integer>> subsets(int[] nums) {
        List<List<Integer>> res = new ArrayList<List<Integer>>();
        List<Integer> path = new ArrayList<Integer>();
        backtrackSubset(res, path, nums, 0);
        return res;
    }

    //iterative way
    public List<List<Integer>> subsetsIterative(int[] nums) {
        List<List<Integer>> res = new ArrayList<List<Integer>>();
        res.add(new ArrayList<Integer>());
        int n = nums.length;
        for (int i = 0; i < n; ++i) {
            int m = res.size();
            for (int j = 0; j < m; ++j) {
                List<Integer> tmp = new ArrayList<Integer>(res.get(j));
                tmp.add(nums[i]);
                res.add(tmp);
            }
        }
        return res;
    }

    //bit manipulation
    //This is the most clever solution that I have seen, the idea is that to give all the possible subsets, we just need to exhaust all the possible combinations of the numbers,
    //and each number has only two possibilities, either in or not in a subset, and this can be represented using abit
    //1 appears once in every two consecutive subsets,2 appear twice in  every four , 3 appear 4 in very eight subsets,
    public List<List<Integer>> subsetsManipulation(int[] nums) {
        int num_subset = 1 << nums.length;//like this sentence very much
        Arrays.sort(nums);
        List<List<Integer>> res = new ArrayList<List<Integer>>();
        for (int i = 0; i < nums.length; ++i) {
            for (int j = 0; j < num_subset; ++j) {
                if (res.size() < j + 1)//interesting question
                    res.add(new ArrayList<Integer>());
                if (((j >> i) & 0x1) != 0)
                    res.get(j).add(nums[i]);
            }
        }
        return res;

    }

    //subsetII
    public void backtrackingWithDup(List<List<Integer>> res, List<Integer> path, int pos, int[] nums) {
        res.add(new ArrayList<Integer>(path));
        for (int i = pos; i < nums.length; ++i) {
            path.add(nums[i]);
            backtrackingWithDup(res, path, i + 1, nums);
            while (i < nums.length - 1 && nums[i] == nums[i + 1])//remove the duplicate;,nums[i] has been used.
                i++;
            path.remove(path.size() - 1);
        }

    }

    public List<List<Integer>> subsetsWithDup(int[] nums) {
        Arrays.sort(nums);//must sort first
        List<List<Integer>> res = new ArrayList<List<Integer>>();
        List<Integer> path = new ArrayList<Integer>();
        backtrackingWithDup(res, path, 0, nums);
        return res;
    }

    public List<List<Integer>> subsetsWithDupIterative(int[] nums) {
        List<List<Integer>> res = new ArrayList<List<Integer>>();
        Arrays.sort(nums);
        int n = 0;
        res.add(new ArrayList<Integer>());
        for (int i = 0; i < nums.length; ++i) {
            int start = (i >= 1 && nums[i] == nums[i - 1]) ? n : 0;//why not i<nums.length-1 && nums[i]==nums[i+1], 针对nums[i+1]的,你这种写法是针对i的，应该使用i-1 和i
            n = res.size();
            for (int j = start; j < n; ++j) {
                List<Integer> tmp = new ArrayList<Integer>(res.get(j));
                tmp.add(nums[i]);
                res.add(tmp);
            }
        }
        return res;
    }


    //167 two sum II input array is sorted
    //you can still use hashmap, but it is a little slow
    public int[] twoSumII(int[] nums, int target) {
        int begin = 0, end = nums.length - 1;
        int[] res = new int[2];
        while (begin < end) {
            if (nums[begin] + nums[end] == target) {
                res[0] = begin;
                res[1] = end;
                break;
            } else if (nums[begin] + nums[end] > target)
                end--;
            else
                begin++;
        }
        return res;
    }

    //15 3sum 3 sum
    //Three sum
    //two sum de hashmap upgrade
    public List<List<Integer>> threeSum(int[] nums) {
        List<List<Integer>> res = new ArrayList<List<Integer>>();
        Arrays.sort(nums);
        int n = nums.length;
        for (int i = 0; i < n - 2; ++i) {  //n-2 not n
            if (i > 0 && nums[i] == nums[i - 1])
                continue;
            int begin = i + 1, end = n - 1;
            while (begin < end) {
                int sum = nums[begin] + nums[i] + nums[end];
                if (sum == 0) {
                    List<Integer> index = new ArrayList<Integer>();
                    index.add(nums[i]);
                    index.add(nums[begin++]);
                    index.add(nums[end--]);
                    res.add(index);
                    while (begin < end && nums[begin] == nums[begin - 1])
                        begin++;
                    while (begin < end && nums[end + 1] == nums[end])
                        end--;
                } else if (sum < 0)
                    begin++;
                else
                    end--;
            }
        }
        return res;
    }

    public List<List<Integer>> fourSum(int[] nums, int target) {
        List<List<Integer>> res = new ArrayList<List<Integer>>();
        Arrays.sort(nums);
        int n = nums.length;
        for (int j = 0; j < n - 3; ++j) {
            if (j > 0 && nums[j] == nums[j - 1])
                continue;
            if (nums[j] + nums[n - 1] + nums[n - 2] + nums[n - 3] < target) continue;
            if (nums[j] + nums[j + 1] + nums[j + 2] + nums[j + 3] > target) break;
            for (int i = j + 1; i < n - 2; ++i) {  //n-2 not n
                if (i > j + 1 && nums[i] == nums[i - 1])
                    continue;
                int begin = i + 1, end = n - 1;
                while (begin < end) {
                    int sum = nums[j] + nums[begin] + nums[i] + nums[end];
                    if (sum == target) {
                        List<Integer> index = new ArrayList<Integer>();
                        index.add(nums[j]);
                        index.add(nums[i]);
                        index.add(nums[begin++]);
                        index.add(nums[end--]);
                        res.add(index);
                        while (begin < end && nums[begin] == nums[begin - 1])
                            begin++;
                        while (begin < end && nums[end + 1] == nums[end])
                            end--;
                    } else if (sum < target)
                        begin++;
                    else
                        end--;
                }
            }
        }
        return res;
    }

    //16 3sum cloest
    public int threeSumClosest(int[] nums, int target) {
        int n = nums.length;
        Arrays.sort(nums);
        int res = nums[0] + nums[1] + nums[2];
        for (int i = 0; i < n - 2; ++i) {
            int begin = i + 1, end = n - 1;
            while (begin < end) {
                int sum = nums[i] + nums[begin] + nums[end];
                if (Math.abs(sum - target) < Math.abs(res - target))
                    res = sum;
                if (sum > target)
                    end--;
                else if (sum < target)
                    begin++;
                else
                    return target;
            }
        }
        return res;
    }

    //31 next permutation

    public void nextPermutation(int[] nums) {
        int n = nums.length;
        int j = n - 1;
        while (j > 0) {
            if (nums[j] > nums[j - 1])
                break;
            j--;
        }
        // 1 2 5 4 3
        //nums is in reverse order
        if (j == 0) {
            Arrays.sort(nums);
            return;
        }
        //find the minimum maximum value
        int key = j - 1;
        int index = j;//iterator
        int res = nums[index];//只能用这种方式，不能用res=nums[index]-nums[key],has bugs
        while (index < n) {
            if (nums[index] > nums[key] && nums[index] < res) {
                res = nums[index];
                j = index;
            }
            index++;
        }
        swap(nums, key, j);
        Arrays.sort(nums, key + 1, n);
    }

    //concise

    public void reverseSort(int[] nums, int begin, int end) {
        if (begin > end)
            return;
        while (begin < end) {
            swap(nums, begin, end);
            begin++;
            end--;
        }
    }

    public void nextPermutationBetter(int[] nums) {
        int n = nums.length;
        if (n < 2)
            return;
        int index = n - 1;
        while (index > 0) {
            if (nums[index] > nums[index - 1])
                break;
            index--;
        }
        if (index == 0) {
            reverseSort(nums, 0, n - 1);
            return;
        } else {
            //find the smallest larger number
            int val = nums[index - 1];
            int j = n - 1;
            while (j >= index) {
                if (nums[j] > val)
                    break;
                j--;
            }
            swap(nums, j, index - 1);
            reverseSort(nums, index, n - 1);
            return;
        }

    }


    public void PrevPermutation(int[] nums) {
        if (nums.length <= 1)
            return;
        int n = nums.length;
        int j = n - 1;
        while (j > 0) {
            if (nums[j] < nums[j - 1])
                break;
        }
        if (j == 0) {
            for (int i = 0; i < n / 2; ++i)
                swap(nums, i, n - i - 1);
            return;
            //Arrays.sort(nums,Collections.reverseOrder());
        }
        int key = j - 1;
        int index = j;
        int maximinValue = nums[index];
        while (index < n) {
            if (nums[index] < key && nums[index] > maximinValue) {
                maximinValue = nums[index];
                j = index;
            }
        }
        swap(nums, key, j);
        Arrays.sort(nums, j, n);//reverse order

    }

    //55 jump game
    public boolean canJump(int[] nums) {
        int n = nums.length;
        int optimalIndex = 0;
        while (optimalIndex < n - 1) {
            if (nums[optimalIndex] == 0)
                return false;
            int saveIndex = optimalIndex;
            int maxReach = nums[saveIndex + 1] + saveIndex + 1;
            for (int j = 1; j <= nums[saveIndex]; ++j) {
                if (maxReach <= nums[saveIndex + j] + saveIndex + j) {
                    maxReach = nums[saveIndex + j] + saveIndex + j;
                    optimalIndex = j;
                    if (maxReach >= n - 1)
                        return true;
                }
            }
        }
        return true;
    }

    //optimal way
    public boolean canJumpBetter(int[] nums) {
        int n = nums.length, maxReach = 0, i = 0;
        for (; i < n && i <= maxReach && maxReach < n - 1; ++i) {
            maxReach = Math.max(maxReach, nums[i] + i);
        }
        return maxReach >= n - 1;
    }


    //34 search for a range, did not pass linkedin
    public int lowerBound(int[] nums, int target) {
        int index = 0, begin = 0, end = nums.length - 1;
        if (nums[end] < target)
            return end + 1;
        while (begin < end) {
            int mid = (end - begin) / 2 + begin;
            if (nums[mid] >= target) {
                end = mid;
                index = end;
            } else {
                begin = mid + 1;
                index = begin;
            }
        }
        return index;
    }

    public int upperBound(int[] nums, int target) {
        int index = 0, begin = 0, end = nums.length - 1;//pos must be 0, think when there is only one element here
        if (nums[end] <= target)
            return end + 1;
        while (begin < end) {
            int mid = (end - begin) / 2 + begin;
            if (nums[mid] > target) {
                end = mid;
                index = end;
            } else {
                begin = mid + 1;
                index = begin;
            }
        }
        return index;
    }

    public int[] searchRange(int[] nums, int target) {
        int[] res = {-1, -1};
        if (nums == null || nums.length == 0)
            return res;
        int index1 = lowerBound(nums, target);
        if (index1 == -1 || index1 == nums.length || nums[index1] != target) {
            return res;
        }
        int index2 = upperBound(nums, target);
        res[0] = index1;
        res[1] = index2 - 1;
        return res;
    }

    //linkedin search range
    //a little hard
    //a lot of corn case
    //The good point here is the lower_bound and the search for (target+1),But I think (target + 1) may caused overflow...
    public int[] searchRangeOneBinarySearch(int[] nums, int target) {
        int[] res = {-1, -1};
        if (nums == null || nums.length == 0)
            return res;
        int begin = 0, end = nums.length - 1;
        while (begin < end) {
            int mid = (end - begin) / 2 + begin;
            if (nums[mid] > target)
                end = mid;
            else if (nums[mid] < target)
                begin = mid + 1;
            else {
                res[0] = begin;
                res[1] = end;
                while (res[0] < mid) {
                    int mid1 = (mid - res[0]) / 2 + res[0];
                    if (nums[mid1] == target)
                        mid = mid1;
                    else
                        res[0] = mid1 + 1;
                }
                while (mid < res[1]) {
                    int mid1 = (res[1] - mid) / 2 + mid;
                    if (nums[mid1] == target)
                        mid = mid1 + 1;
                    else
                        res[1] = mid1;
                }
                if (nums[res[1]] != target)
                    res[1]--;//does not equal to target
                break;
            }
        }
        //only one element in the array.
        if (res[0] == -1 && nums[begin] == target) {
            res[0] = res[1] = begin;
        }
        return res;

    }

    //289 game of life

    public int check(int[][] save, int x, int y) {
        int cnt = 0;
        for (int i = x - 1; i <= x + 1; ++i) {
            if (i < 0 || i >= save.length) continue;
            for (int j = y - 1; j <= y + 1; ++j) {
                if (j >= 0 && j < save[0].length) {
                    if (!(x == i && y == j)) {
                        if (save[i][j] == 1)
                            cnt++;
                    }
                }
            }
        }
        return cnt;
    }

    public void gameOfLife(int[][] board) {
        if (board.length == 0 || board[0].length == 0)
            return;
        int m = board.length, n = board[0].length;
        int[][] save = new int[m][n];
        for (int i = 0; i < m; ++i) {
            save[i] = board[i].clone();
        }
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                int cnt = check(save, i, j);
                if (board[i][j] == 1 && (cnt < 2 || cnt > 3))
                    board[i][j] = 0;
                else if (board[i][j] == 0 && cnt == 3)
                    board[i][j] = 1;
            }
        }
    }

    //interesting question
    //- 00  dead (next) <- dead (current)
    //- 01  dead (next) <- live (current)
    //- 10  live (next) <- dead (current)
    //- 11  live (next) <- live (current)
    //highest vote algo
    /*
    To get the current state, simply do

    board[i][j] & 1
    To get the next state, simply do
    board[i][j] >> 1
     */
    public void gameOfLifeOptimal(int[][] board) {
        if (board == null || board.length == 0) return;
        int m = board.length, n = board[0].length;

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                int lives = liveNeighbors(board, m, n, i, j);

                // In the beginning, every 2nd bit is 0;
                // So we only need to care about when will the 2nd bit become 1.
                if (board[i][j] == 1 && lives >= 2 && lives <= 3) {
                    board[i][j] = 3; // Make the 2nd bit 1: 01 ---> 11
                }
                if (board[i][j] == 0 && lives == 3) {
                    board[i][j] = 2; // Make the 2nd bit 1: 00 ---> 10
                }
            }
        }

        for (int i = 0; i < m; i++) {
            for (int j = 0; j < n; j++) {
                board[i][j] >>= 1;  // Get the 2nd state.
            }
        }
    }

    public int liveNeighbors(int[][] board, int m, int n, int i, int j) {
        int lives = 0;
        for (int x = Math.max(i - 1, 0); x <= Math.min(i + 1, m - 1); x++) {
            for (int y = Math.max(j - 1, 0); y <= Math.min(j + 1, n - 1); y++) {
                lives += board[x][y] & 1;
            }
        }
        lives -= board[i][j] & 1;
        return lives;
    }

    //hard part
    //33 search in rotated array
    public int searchInSortedArray(int[] nums, int target) {
        int begin = 0, end = nums.length - 1;
        while (begin < end) {
            int mid = (end - begin) / 2 + begin;
            if (nums[mid] == target)
                return mid;
            if (nums[mid] > nums[end]) {
                if (nums[mid] > target && target >= nums[begin])
                    end = mid;
                else
                    begin = mid + 1;
            } else if (nums[mid] < nums[end]) {
                if (nums[mid] < target && target <= nums[end])
                    begin = mid + 1;
                else
                    end = mid;
            }
        }
        return nums[begin] == target ? begin : -1;
    }
    //81 search in rotated array II;
    //just add  else end--;

    public int searchInSortedArrayAnotherVersion(int[] nums, int target) {
        int n = nums.length;
        if (n == 0)
            return -1;
        int lo = 0, hi = n - 1;
        while (lo < hi) {
            int mid = (hi - lo) / 2 + lo;
            if (nums[mid] > nums[hi])
                lo = mid + 1;
            else
                hi = mid;
        }
        int rot = lo;
        lo = 0;
        hi = n - 1;
        while (lo < hi) {
            int mid = (hi - lo) / 2 + lo;
            int readlmid = (rot + mid) % n;
            if (nums[readlmid] == target)
                return readlmid;
            else if (nums[readlmid] > target)
                hi = mid;
            else
                lo = mid + 1;
        }
        int real = (rot + lo) % n;
        return nums[real] == target ? real : -1;
    }


    //287
    //O(nlogn) sort and find
    //set o(n) but use extra space
    //two pointers
    public int findDuplicate(int[] nums) {
        if (nums.length > 1) {
            int slow = nums[0];
            int fast = nums[nums[0]];
            while (slow != fast) {
                slow = nums[slow];
                fast = nums[nums[fast]];
            }
            fast = 0;
            while (fast != slow) {
                fast = nums[fast];
                slow = nums[slow];
            }
            return slow;
        }
        return -1;
    }


    //128 longest consecutive sequence
    public int longestConsecutive(int[] nums) {
        Set<Integer> set = new HashSet<Integer>();
        int maxLength = 0;
        for (int x : nums)
            set.add(x);
        for (int x : nums) {
            if (!set.contains(x))
                continue;
            set.remove(x);
            int pre = x - 1;
            int next = x + 1;
            while (set.contains(pre)) {
                set.remove(pre);
                pre--;
            }
            while (set.contains(next)) {
                set.remove(next);
                next++;
            }
            maxLength = Math.max(maxLength, next - pre - 1);
        }
        return maxLength;
    }

    //57 insert interval
    public List<Interval> insert(List<Interval> intervals, Interval newInterval) {
        List<Interval> res = new ArrayList<Interval>();
        int i = 0, n = intervals.size();
        while (i < n) {
            if (intervals.get(i).end < newInterval.start)
                res.add(intervals.get(i++));
            else
                break;
        }
        while (i < n && intervals.get(i).start <= newInterval.end) {
            newInterval.start = Math.min(intervals.get(i).start, newInterval.start);
            newInterval.end = Math.max(intervals.get(i).end, newInterval.end);
            i++;
        }
        res.add(newInterval);
        while (i < n) {
            res.add(intervals.get(i++));
        }
        return res;
    }

    //56 merge interval
    public List<Interval> merge(List<Interval> intervals) {
        List<Interval> res = new ArrayList<Interval>();
//        intervals.sort(new Comparator<Interval>() {
//            public int compare(Interval o1, Interval o2) {
//                return Integer.compare(o1.start,o2.start);
//            }
//        });
        intervals.sort((Interval o1, Interval o2) -> o1.start - o2.start);
        int i = 1, n = intervals.size();
        if (n == 0)
            return res;
        res.add(intervals.get(0));
        while (i < n) {
            if (res.get(res.size() - 1).end >= intervals.get(i).start) {
                res.get(res.size() - 1).end = Math.max(res.get(res.size() - 1).end, intervals.get(i).end);
            } else
                res.add(intervals.get(i));
            i++;
        }
        return res;
    }

    //42 trapping rain water
    public int trap(int[] height) {
        int begin = 0, n = height.length;
        int res = 0, minHeight = 0;
        int[] left = new int[n];
        int[] right = new int[n];
        for (int i = 1; i < n; ++i) {
            left[i] = Math.max(left[i - 1], height[i - 1]);
            right[n - 1 - i] = Math.max(right[n - i], height[n - i]);
        }
        for (int i = 0; i < n; ++i) {
            minHeight = Math.min(left[i], right[i]);
            res += Math.max(0, minHeight - height[i]);
        }
        return res;
    }

    //save more space
    public int trapSaveSpace(int[] height) {
        int begin = 0, n = height.length, end = n - 1;
        int res = 0, leftHeight = 0, rightHeight = 0;
        while (begin < end) {
            leftHeight = Math.max(leftHeight, height[begin]);
            rightHeight = Math.max(rightHeight, height[end]);
            int minHeight = Math.min(leftHeight, rightHeight);
            if (height[begin] < height[end]) {
                res += Math.max(0, minHeight - height[begin++]);
            } else {
                res += Math.max(0, minHeight - height[end--]);
            }
        }
        return res;
    }

    //more concise
    public int trapSaveSpaceConcise(int[] height) {
        int begin = 0, n = height.length, end = n - 1;
        int res = 0, leftHeight = 0, rightHeight = 0;
        while (begin < end) {
            if (height[begin] < height[end]) {
                leftHeight = Math.max(leftHeight, height[begin]);
                res += Math.max(0, leftHeight - height[begin++]);
            } else {
                rightHeight = Math.max(rightHeight, height[end]);
                res += Math.max(0, rightHeight - height[end--]);
            }
        }
        return res;
    }

    //41 first missing positive
    //sort and find
    public int firstMissingPositive(int[] nums) {
        Arrays.sort(nums);
        int n = nums.length;
        int cnt = 1;
        for (int i = 0; i < n; ++i) {
            if (nums[i] < 1 || (i > 0 && nums[i - 1] == nums[i]))//duplicate
                continue;
            if (nums[i] != cnt)
                return cnt;
            cnt++;
        }
        return cnt;
    }

    //O(N) time and save space
    //coner case 1,2,3
    //conner case 0,1,1,2,2//duplicate
    public int firstMissingPositiveSaveSpace(int[] nums) {
        int n = nums.length;
        for (int i = 0; i < n; ++i) {
            while (nums[i] >= 1 && nums[i] <= n && nums[nums[i] - 1] != nums[i]) {
                swap(nums, nums[i] - 1, i);
            }
        }
        for (int i = 0; i < n; ++i) {
            if (nums[i] != i + 1)
                return i + 1;
        }
        return n + 1;
    }

    //84 largest rectangle in histogram
    public int largestRectangleArea(int[] heights) {
        Stack<Integer> stk = new Stack<>();//to store the index
        int n = heights.length, res = 0;
        for (int i = 0; i <= n; ++i) {
            int h = (i == n ? 0 : heights[i]);
            while (!stk.isEmpty() && heights[stk.peek()] > h) {
                int height = heights[stk.pop()];
                int index = stk.isEmpty() ? i : i - 1 - stk.peek();//stk.peek() is also not in the range, and i is not in the range
                res = Math.max(res, height * index);
            }
            stk.push(i);
        }
        return res;
    }

    // second; you have to handle the 1,2,3,4,5 case
    public int largestRectangleArea2(int[] heights) {
        Stack<Integer> stk = new Stack<>();//to store the index
        int n = heights.length, res = 0;
        for (int i = 0; i < n; ++i) {
            while (!stk.isEmpty() && heights[stk.peek()] > heights[i]) {
                int height = heights[stk.pop()];
                int index = stk.isEmpty() ? i : i - 1 - stk.peek();
                ;
                res = Math.max(res, height * index);
            }
            stk.push(i);
        }
        while (!stk.isEmpty()) {
            int height = heights[stk.pop()];
            int index = stk.isEmpty() ? n : n - 1 - stk.peek();
            res = Math.max(res, height * index);
        }
        return res;
    }

    //85 maximal rectangle
    public int maximalRectangle(char[][] matrix) {
        if (matrix.length == 0 || matrix[0].length == 0)
            return 0;
        int res = 0, m = matrix.length, n = matrix[0].length;
        int[][] dp = new int[m][n];
        for (int i = 0; i < n; ++i) {
            dp[0][i] = matrix[0][i] == '1' ? 1 : 0;
        }
        res = Math.max(res, largestRectangleArea(dp[0]));
        for (int i = 1; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                dp[i][j] = matrix[i][j] == '0' ? 0 : 1 + dp[i - 1][j];
            }
            res = Math.max(res, largestRectangleArea(dp[i]));
        }
        return res;
    }

    //best time to but and sell stokc IIi
    // lt 123
    public int maxProfitIIi(int[] prices) {
        //two arrays
        int n = prices.length;
        int[] first = new int[n + 1];
        int[] second = new int[n + 1];
        int val = Integer.MAX_VALUE;
        int res = 0;
        for (int i = 1; i <= n; ++i) {
            val = Math.min(val, prices[i - 1]);
            first[i] = Math.max(first[i - 1], prices[i - 1] - val);//concern with last statement, you can use n+1
        }
        val = 0;
        for (int i = n - 1; i >= 0; --i) {
            val = Math.max(val, prices[i]);
            second[i] = Math.max(second[i + 1], val - prices[i]);
            res = Math.max(res, first[i] + second[i]);
        }
        return res;
    }

    //45 jump gameII
    public int jumpII(int[] nums) {
        if (nums.length <= 1)
            return 0;
        int n = nums.length;
        int cnt = 0, optimalIndex = 0;
        while (optimalIndex < n - 1) {
            int index = optimalIndex + 1, maxReach = nums[index] + index;
            for (int j = 1; j <= nums[optimalIndex]; ++j) {
                if (optimalIndex + j >= n - 1) {
                    cnt++;
                    return cnt;
                }
                if (nums[optimalIndex + j] + j + optimalIndex >= maxReach) {
                    index = optimalIndex + j;
                    maxReach = nums[index] + index;
                }
            }
            cnt++;
            optimalIndex = index;
        }
        return cnt;
    }

    //median of two sorted arrays
    //The overall run time complexity should be O(log (m+n)).
    public double findkth(int[] nums1, int start1, int[] nums2, int start2, int k) {
        if (nums1.length - start1 > nums2.length - start2)
            return findkth(nums2, start2, nums1, start1, k);
        if (start1 == nums1.length)
            return nums2[start2 + k - 1];//in case of index boundary overflow
        if (k == 1)
            return Math.min(nums1[start1], nums2[start2]);
        int pa = Math.min(k / 2, nums1.length - start1);
        int pb = k - pa;
        if (nums1[start1 + pa - 1] < nums2[start2 + pb - 1])
            return findkth(nums1, start1 + pa, nums2, start2, k - pa);
        else if (nums1[start1 + pa - 1] > nums2[start2 + pb - 1])
            return findkth(nums1, start1, nums2, start2 + pb, k - pb);
        return nums1[start1 + pa - 1];//already find the median
    }

    public double findMedianSortedArrays(int[] nums1, int[] nums2) {
        if (nums1.length == 0 && nums2.length == 0)
            return 0.0;
        int m = nums1.length, n = nums2.length;
        int l = (m + n + 1) >> 1;
        int r = (m + n + 2) >> 1;
        return (findkth(nums1, 0, nums2, 0, l) + findkth(nums1, 0, nums2, 0, r)) / 2.0;
    }
    //merge two sorted arrays O(m+n)
    //and then find

    //127 word ladder
    // yuu can spped up, think about it
    public int ladderLength(String beginWord, String endWord, Set<String> wordList) {
        //remove the used word
        int res = 2;
        if (beginWord == endWord)
            return 1;
        wordList.remove(beginWord);
        wordList.remove(endWord);
        Queue<String> q = new LinkedList<>();
        q.offer(beginWord);
        while (!q.isEmpty()) {
            int size = q.size();
            while (size-- > 0) {
                StringBuffer cur = new StringBuffer(q.poll());
                for (int i = 0; i < cur.length(); ++i) {
                    char p = cur.charAt(i);
                    for (char c = 'a'; c <= 'z'; ++c) {
                        cur.setCharAt(i, c);
                        if (cur.toString().equals(endWord))
                            return res;
                        if (wordList.contains(cur.toString())) {
                            q.offer(cur.toString());
                            wordList.remove(cur.toString());
                        }
                    }
                    cur.setCharAt(i, p);
                }
            }
            res++;
        }
        return 0;
    }


    //485 max consecutive ones
    public int findMaxConsecutiveOnes(int[] nums) {
        int res = 0, n = nums.length, begin = 0, end = 0;
        while (end < n) {
            while (end < n && nums[end] == 0)
                end++;
            begin = end;
            while (end < n && nums[end] == 1)
                end++;
            res = Math.max(res, end - begin);
        }

        return res;
    }

    //487 max consecutive ones
    public int findMaxConsecutiveOnesII(int[] nums) {
        int res = 0, n = nums.length, begin = 0, end = 0;
        List<Interval> inter = new ArrayList<>();
        while (end < n) {
            while (end < n && nums[end] == 0)
                end++;
            begin = end;
            while (end < n && nums[end] == 1)
                end++;
            inter.add(new Interval(begin, end - 1));
            res = Math.max(res, end - begin);
        }
        if (res == n)
            return n;
        int m = inter.size();
        boolean flipped = false;
        for (int i = 1; i < m; ++i) {
            if (inter.get(i).start == inter.get(i - 1).end + 2) {

                int newLength = inter.get(i).end - inter.get(i).start + 1 + inter.get(i - 1).end - inter.get(i - 1).start + 2;
                if (newLength > res) {
                    res = newLength;
                    flipped = true;// flip should here, not in the line 2692, which is the last case I can not pass
                }
            }
        }
        return flipped ? res : res + 1;
    }


    //163 missing ranges
    //test case [-2147483648,-2147483648,0,2147483647,2147483647]
    //-2147483648
    //  2147483647
    public List<String> findMissingRanges(int[] nums, int lower, int upper) {
        int n = nums.length, index = 1;
        long start = lower, end = lower;
        List<String> res = new ArrayList<>();
        if (n == 0) {
            res.add(lower == upper ? "" + lower : lower + "->" + upper);
            return res;
        }
        if (nums[0] != lower) {
            res.add(nums[0] == lower + 1 ? "" + lower : lower + "->" + (nums[0] - 1));

        }
        while (index < n) {
            if ((long) nums[index] != (long) (nums[index - 1]) + 1) {
                start = (long) nums[index - 1] + 1;
                end = (long) nums[index] - 1;
                if (start <= end)
                    res.add(start == end ? "" + start : start + "->" + end);
            }
            index++;
        }
        if (nums[n - 1] != upper) {
            res.add(nums[n - 1] == upper - 1 ? "" + upper : (nums[n - 1] + 1) + "->" + upper);
        }
        return res;
    }

    //better ways
    public List<String> findMissingRangesbetter(int[] a, int low, int high) {
        //there is no need to compare the neighbour element
        long lo = (long) low;
        long hi = (long) high;
        List<String> res = new ArrayList<>();
        for (int num : a) {
            long justbelow = (long) num - 1;
            if (justbelow == lo)
                res.add(justbelow + "");
            else if (justbelow > lo)
                res.add(lo + "->" + justbelow);
            lo = (long) num + 1;
        }
        if (lo == hi)
            res.add(lo + "");
        else if (lo < hi)
            res.add(lo + "->" + hi);
        return res;
    }


    //370 range addition
    public int[] getModifiedArray(int length, int[][] updates) {
        int[] res = new int[length];
        for (int[] p : updates) {
            res[p[0]] += p[2];
            if (p[1] < length - 1)
                res[p[1] + 1] -= p[2];
        }
        int sum = 0;
        for (int i = 1; i < length; ++i) {
            res[i] += res[i - 1];
        }
        return res;
    }

    //436 find the right interval
    public int binarySearch(List<Unit> res, int target) {
        int n = res.size() - 1, begin = 0;
        while (begin < n) {
            int mid = (n - begin) / 2 + begin;
            if (res.get(mid).a.start >= target)
                n = mid;
            else
                begin = mid + 1;
        }
        return res.get(begin).a.start >= target ? res.get(begin).index : -1;
    }

    public int[] findRightInterval(Interval[] intervals) {
        List<Unit> res = new ArrayList<>();
        int n = intervals.length;
        for (int i = 0; i < n; ++i) {
            res.add(new Unit(intervals[i], i));
        }
        res.sort((Unit aa, Unit bb) -> aa.a.start - bb.a.start);
        int[] ans = new int[n];
        for (int i = 0; i < n; ++i) {
            ans[i] = binarySearch(res, intervals[i].end);
        }
        return ans;

    }

    public int findPoisonedDuration(int[]timeSeries,int duration){
        List<Interval>res=new ArrayList<>();
        for(int x:timeSeries){
            res.add(new Interval(x,x+duration-1));
        }
        List<Interval>ans=merge(res);
        int cnt=0;
        for(Interval interval:ans){
            cnt+=interval.end-interval.start+1;
        }
        return cnt;
    }

    //475. Heaters
    public int findRadius(int[] houses, int[] heaters) {
        Arrays.sort(houses);
        Arrays.sort(heaters);
        int res = Integer.MIN_VALUE;
        int j = 0;
        for (int i = 0; i < houses.length; ++i) {
            while (j < heaters.length - 1 && Math.abs(heaters[j] - houses[i]) >= Math.abs(heaters[j + 1] - houses[i]))
                j++;
            res = Math.max(res, Math.abs(heaters[j] - houses[i]));
        }
        return res;
    }

    public int[] nextGreaterElement(int[] findNums, int[] nums) {
        List<Integer> res = new ArrayList<>();
        int m = findNums.length, n = nums.length;
        for (int i = 0; i < m; ++i) {
            int j = 0;
            for (; j < n; ++j) {
                if (findNums[i] == nums[j])
                    break;
            }
            boolean flag = true;
            for (int k = j + 1; k < n; ++k) {
                if (nums[k] > findNums[i]) {
                    res.add(nums[k]);
                    flag = false;
                    break;
                }
            }
            if (flag)
                res.add(-1);
        }
        int[] ans = new int[res.size()];
        for (int i = 0; i < ans.length; ++i)
            ans[i] = res.get(i);
        return ans;
    }

    public int[] nextGreaterElements(int[] nums) {
        int n = nums.length;
        int[] res = new int[n];
        for (int i = 0; i < n; ++i) {
            int j = i + 1;
            boolean flag = true;
            for (; j < n + i; ++j) {
                if (nums[j % n] > nums[i]) {
                    res[i] = nums[j % n];
                    flag = false;
                    break;

                }
            }
            if (flag)
                res[i] = -1;

        }
        return res;
    }

    //stack, cycle

    public int[] nextGreaterElementsByStack(int[] nums) {
        int n = nums.length;
        int[] res = new int[n];
        Arrays.fill(res, -1);
        Stack<Integer> stk = new Stack<>();
        for (int i = 0; i < 2 * n; ++i) {
            int num = nums[i % n];
            while (!stk.isEmpty() && nums[stk.peek()] < num)
                res[stk.pop()] = num;
            if (i < n)
                stk.push(i);
        }
        return res;
    }

    public int[] nextGreaterElementsByStackNoCycle(int[] findNums, int[] nums) {
        Map<Integer, Integer> map = new HashMap<>();
        Stack<Integer> stk = new Stack<>();
        for (int num : nums) {
            while (!stk.isEmpty() && stk.peek() < num)
                map.put(stk.pop(), num);
            stk.push(num);
        }
        for (int i = 0; i < findNums.length; ++i) {
            findNums[i] = map.getOrDefault(findNums[i], -1);
        }
        return findNums;
    }


    public int[] findDiagonalOrder(int[][] matrix) {
        List<List<Integer>> res = new ArrayList<>();
        if (matrix.length == 0 || matrix[0].length == 0)
            return new int[0];
        int m = matrix.length, n = matrix[0].length;
        for (int j = 0; j < n; ++j) {
            int i = 0;
            List<Integer> path = new ArrayList<>();
            for (int k = j; k >= 0 && i < m; --k)
                path.add(matrix[i++][k]);
            res.add(path);
        }
        for (int i = 1; i < m; ++i) {
            int j = n - 1;
            List<Integer> path = new ArrayList<>();
            for (int k = i; k < m && j >= 0; ++k)
                path.add(matrix[k][j--]);
            res.add(path);
        }
        for (int i = 0; i < res.size(); ++i) {
            if (i % 2 == 0)
                Collections.reverse(res.get(i));
        }
        int[] ans = new int[m * n];
        int index = 0;
        for (int i = 0; i < res.size(); ++i) {
            for (int x : res.get(i))
                ans[index++] = x;
        }
        return ans;

    }

    //302 Smallest Rectangle Enclosing Black Pixels
    //binary search

    public boolean check(char[][] image, int mid, boolean isCol) {
        if (isCol) {
            for (int i = 0; i < image.length; ++i) {
                if (image[i][mid] == '1')
                    return true;
            }
            return false;
        } else {
            for (int i = 0; i < image[0].length; ++i) {
                if (image[mid][i] == '1')
                    return true;
            }
            return false;
        }
    }

    //can check begin,but can't check end, so you 'd better begin again before return
    public int binarySearch(char[][] image, int begin, int end, boolean isCol, boolean isLower) {
        while (begin < end) {
            int mid = (end - begin) / 2 + begin;
            if (isLower) {
                if (check(image, mid, isCol))
                    end = mid;
                else
                    begin = mid + 1;
            } else {
                if (!check(image, mid, isCol))
                    end = mid;
                else
                    begin = mid + 1;
            }
        }
        return !check(image, begin, isCol) ? begin - 1 : begin;
    }

    public int minArea(char[][] image, int x, int y) {
        if (image.length == 0 || image[0].length == 0)
            return 0;
        int m = image.length, n = image[0].length;
        int left = binarySearch(image, 0, y, true, true);
        int right = binarySearch(image, y, n - 1, true, false);
        int upper = binarySearch(image, 0, x, false, true);
        int lower = binarySearch(image, x, m - 1, false, false);
        //right=Math.min(right,n-1);
        //lower=Math.min(lower,m-1);
        return (right - left + 1) * (lower - upper + 1);
    }

    //Sort elements by frequency
    //you can also use bst
    //you can also use count sort;//
    public void sortByFrequency(int[] nums) {
        Map<Integer, int[]> map = new HashMap<>();
        int n = nums.length;
        for (int i = 0; i < n; ++i) {
            if (map.containsKey(nums[i])) {
                int[] tmp = map.get(nums[i]);
                tmp[0]++;
                map.put(nums[i], tmp);
            } else {
                map.put(nums[i], new int[]{1, i});
            }
        }
        //struct to push all
        List<ValueWtihFrequency> li = new ArrayList<>();
        for (Map.Entry<Integer, int[]> entry : map.entrySet()) {
            li.add(new ValueWtihFrequency(entry.getKey(), entry.getValue()[0], entry.getValue()[1]));
        }
        li.sort(new Comparator<ValueWtihFrequency>() {
            @Override
            public int compare(ValueWtihFrequency o1, ValueWtihFrequency o2) {
                if (o1.count == o2.count) {
                    if (o1.index == o2.index)
                        return 0;
                    else if (o1.index > o2.index)
                        return 1;
                    else
                        return -1;
                } else if (o1.count > o2.count)
                    return -1;
                else
                    return 1;
            }
        });

        for (ValueWtihFrequency f : li) {
            int count = f.count;
            while (count-- > 0)
                System.out.println(f.val);
        }
    }

    public void merge(int[] nums, int start, int mid, int end, int[] count) {
        int[] res = new int[end - start + 1];
        int i = start, j = mid + 1, index = 0;
        while (i <= mid && j <= end) {
            if (nums[i] > nums[j]) {
                res[index++] = nums[j++];
                count[0] += mid - i + 1;
            } else
                res[index++] = nums[i++];
        }
        while (i <= mid) {
            res[index++] = nums[i++];
        }
        while (j <= end)
            res[index++] = nums[j++];
        for (int k = 0; k < index; ++k)
            nums[k + start] = res[k];
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

    //dan diao queue
    public int[] maxSlidingWindow(int[] nums, int k) {
        int n = nums.length;
        if (n == 0 || k == 0 || n < k) return new int[0];
        int[] res = new int[n - k + 1];
        int index = 0;
        Deque<Integer> q = new LinkedList<>();
        for (int i = 0; i < n; ++i) {

            while (!q.isEmpty() && nums[q.peekLast()] < nums[i]) {
                q.pollLast();
            }
            q.offer(i);
            if (i >= k - 1)
                res[index++] = nums[q.peekFirst()];
            if (!q.isEmpty() && q.peekFirst() <= i - k + 1)
                q.pollFirst();
        }
        return res;
    }


    //lint code Maximum Subarray II
    public int getMax(ArrayList<Integer> nums, int begin, int end) {
        int secMaxSum = Integer.MIN_VALUE;
        int sum = 0;
        for (int i = begin; i <= end && i < nums.size(); ++i) {
            sum += nums.get(i);
            if (secMaxSum < sum)
                secMaxSum = sum;
            if (sum < 0)
                sum = 0;
        }
        return secMaxSum;
    }

    public int maxTwoSubArrays(ArrayList<Integer> nums) {
        // write your code
        int maxSum = Integer.MIN_VALUE;
        int start = 0, end = 0, s = 0, sum = 0, n = nums.size();
        for (int i = 0; i < n; ++i) {
            sum += nums.get(i);
            if (maxSum < sum) {
                start = s;
                end = i;
                maxSum = sum;
            }
            if (sum < 0) {
                sum = 0;
                s = i + 1;
            }
        }
        int val2 = Math.max(getMax(nums, 0, start - 1), getMax(nums, end + 1, n - 1)) + maxSum;//
        if (start != end) {
            int minValue = Integer.MAX_VALUE;
            for (int i = start; i <= end; ++i) {
                minValue = Math.min(minValue, nums.get(i));
            }
            int val1 = maxSum - minValue;//-minium value

            return Math.max(val1, Math.max(val2, maxSum));
        } else {
            return val2;
        }
    }

    //Find the two repeating elements in a given array
    public void printTwoRepeating(int arr[]) {
        int n = arr.length;
        for (int i = 0; i < n; ++i) {
            int val = Math.abs(arr[i]) - 1;
            if (arr[val] < 0)
                System.out.println(val + 1);
            arr[val] = -arr[val];
        }
    }

    public int equilibrium(int arr[]) {
        int n = arr.length;
        int[] res = new int[n];
        res[0] = arr[0];
        for (int i = 1; i < n; ++i)
            res[i] = res[i - 1] + arr[i];
        for (int i = n - 1; i >= 0; --i) {
            int hiSum = i == n - 1 ? 0 : res[n - 1] - res[i];
            int lowSum = i == 0 ? 0 : res[i - 1];
            if (hiSum == lowSum)
                System.out.println(i);
        }
        return 0;
    }


    public int cycleSort(int[] nums) {
        int writes = 0;
        int n = nums.length;
        for (int i = 0; i < n - 1; ++i) {
            int item = nums[i];

            //find where to put the item
            int pos = i;
            for (int j = i + 1; j < n; ++j) {
                if (nums[j] < item)
                    pos++;
            }

            //if the item is already there, this is not a cycle
            if (pos == i)
                continue;

            //otherwise, put the item thre or right after any duplicate
            while (item == nums[pos])
                pos++;
            if (item != nums[pos]) {
                int tmp = item;
                item = nums[pos];
                nums[pos] = tmp;
                writes++;
            }

            //rotate the rest of the cycle
            while (pos != i) {
                pos = i;
                for (int j = i + 1; j < n; ++j) {
                    if (nums[j] < item)
                        pos++;
                }


                while (item == nums[pos]) {
                    pos++;
                }
                if (item != nums[pos]) {
                    int tmp = item;
                    item = nums[pos];
                    nums[pos] = tmp;
                    writes++;
                }
            }
        }
        System.out.println(writes);
        return writes;
    }

    //Check if array elements are consecutive
    //three kinds of ways, sort , (get min ,max, and use visited array)
    public int[] getMinMax(int[] nums) {
        int minValue = Integer.MAX_VALUE;
        int maxValue = Integer.MIN_VALUE;
        for (int x : nums) {
            minValue = Math.min(minValue, x);
            maxValue = Math.max(maxValue, x);
        }
        return new int[]{minValue, maxValue};
    }

    public boolean areConsecutiveWithVisitedArray(int[] nums) {
        int n = nums.length;
        int[] minMax = getMinMax(nums);
        if (minMax[1] - minMax[0] != n - 1)
            return false;
        boolean[] vis = new boolean[n];
        for (int i = 0; i < n; ++i) {
            if (vis[nums[i] - minMax[0]])
                return false;
            vis[nums[i] - minMax[0]] = true;
        }
        return true;
    }

    //without visited array
    //can only deal with positive integer
    public boolean areConsecutive(int[] nums) {
        int n = nums.length;
        int[] minMax = getMinMax(nums);
        if (minMax[1] - minMax[0] != n - 1)
            return false;
        for (int i = 0; i < n; ++i) {
            int val = Math.abs(nums[i]) - minMax[0];
            if (nums[val] < 0)
                return false;
            nums[val] = -nums[val];
        }
        return true;
    }

    //Find the Minimum length Unsorted Subarray, sorting which makes the complete array sorted
    /*
    Solution:
    1) Find the candidate unsorted subarray
    a) Scan from left to right and find the first element which is greater than the next element. Let s be the index of such an element. In the above example 1, s is 3 (index of 30).
    b) Scan from right to left and find the first element (first in right to left order) which is smaller than the next element (next in right to left order). Let e be the index of such an element. In the above example 1, e is 7 (index of 31).

    2) Check whether sorting the candidate unsorted subarray makes the complete array sorted or not. If not, then include more elements in the subarray.
    a) Find the minimum and maximum values in arr[s..e]. Let minimum and maximum values be min and max. min and max for [30, 25, 40, 32, 31] are 25 and 40 respectively.
    b) Find the first element (if there is any) in arr[0..s-1] which is greater than min, change s to index of this element. There is no such element in above example 1.
    c) Find the last element (if there is any) in arr[e+1..n-1] which is smaller than max, change e to index of this element. In the above example 1, e is changed to 8 (index of 35)

    3) Print s and e.
     */

    public void printUnsorted(int[] nums) {
        int n = nums.length, s = 0, e = n - 1;
        for (; s < n - 1; ++s) {
            if (nums[s] > nums[s + 1])
                break;
        }
        if (s == n - 1)
            System.out.println("the complete array is sorted");
        for (; e > 0; --e) {
            if (nums[e] < nums[e - 1])
                break;
        }

        //find minimum and maximum in the interval
        int maxValue = nums[s], minValue = nums[s];
        for (int i = s + 1; i <= e; ++i) {
            if (nums[i] > maxValue)
                maxValue = nums[i];
            if (nums[i] < minValue)
                minValue = nums[i];
        }

        //steps Three
        for (int i = 0; i < s; ++i) {
            if (nums[i] > minValue) {
                s = i;
                break;
            }
        }
        for (int i = n - 1; i >= e + 1; --i) {
            if (nums[i] < maxValue) {
                e = i;
                break;
            }
        }

        // step 3 of above algo
        System.out.println("The unsorted subarray which makes the given array sorted lies between the index " + s + "  " + e);
    }

    public int maxIndexDiff(int[] nums) {
        int n = nums.length;
        if (n == 0)
            return -1;
        int[] leftMin = new int[n];
        int[] rightMax = new int[n];
        leftMin[0] = nums[0];
        rightMax[n - 1] = nums[n - 1];
        for (int i = 1; i < n; ++i) {
            leftMin[i] = Math.min(leftMin[i - 1], nums[i]);
            rightMax[n - i - 1] = Math.max(rightMax[n - i], nums[n - i - 1]);
        }

        int i = 0, j = 0, maxDiff = -1;
        while (i < n && j < n) {
            if (leftMin[i] < rightMax[j]) {
                maxDiff = Math.max(maxDiff, j - i);
                j++;
            } else
                i++;
        }
        return maxDiff;
    }


    //Find whether an array is subset of another array | Added Method 3
    //brute force
    //sort and binary search
    //sort and merge;// two pointers
    //hashmap: you can use two hashmap to handle the duplicate case
    public boolean isSubset(int[] nums, int[] nums1) {
        int m = nums.length, n = nums1.length;
        Arrays.sort(nums);
        Arrays.sort(nums1);
        int i = 0, j = 0;
        while (i < m && j < n) {
            if (nums[i] < nums1[j])
                i++;
            else if (nums[i] == nums1[j]) {
                i++;
                j++;
            } else if (nums[i] > nums1[j]) {
                return false;
            }
        }
        return i < n;
    }


    //LTE the last example
//    public int findMaximizedCapital(int k, int W, int[] Profits, int[] Capital) {
//        int n=Profits.length;
//        List<Project>li=new ArrayList<>();
//        for(int i=0;i<n;++i)
//            li.add(new Project(Profits[i],Capital[i],false));
//        li.sort((Project p1,Project p2)-> p2.profit-p1.profit);
//        for(int i=0;i<k;++i){
//            for(int j=0;j<n;++j){
//                if(!li.get(j).used && W>=li.get(j).capital){
//                    li.get(j).used=true;
//                    W+=li.get(j).profit;
//                    break;
//                }
//            }
//        }
//        return W;
//
//    }

    public int findMaximizedCapital(int k, int W, int[] Profits, int[] Capital) {
        PriorityQueue<int[]> pqCap = new PriorityQueue<>((a, b) -> a[0] - b[0]);
        PriorityQueue<int[]> pqPro = new PriorityQueue<>((a, b) -> b[1] - a[1]);
        int n = Profits.length;
        for (int i = 0; i < n; ++i) {
            pqCap.offer(new int[]{Capital[i], Profits[i]});
        }

        for (int i = 0; i < k; ++i) {
            while (!pqCap.isEmpty() && pqCap.peek()[0] <= W)
                pqPro.add(pqCap.poll());
            if (pqPro.isEmpty())
                break;
            W += pqPro.poll()[1];
        }
        return W;
    }
}
