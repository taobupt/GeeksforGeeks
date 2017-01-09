package array;

import java.util.*;

/**
 * Created by tao on 1/9/17.
 */
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
    public int maxSubArray(int[] nums) {
        int sum = 0;
        int res = Integer.MIN_VALUE;
        for (int x : nums) {
            if (sum >= 0) {
                sum += x;
            } else
                sum = x;
            res = Math.max(res, sum);
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
                right = sum;
            }
        }
        return leftsum + rightsum;
    }



}
