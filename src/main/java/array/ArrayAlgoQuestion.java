package array;

import sun.rmi.server.InactiveGroupException;

import java.util.*;

/**
 * Created by tao on 1/9/17.
 */


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
    public String moveCharacter(String nums) {
        int j = 0;
        int n = nums.length();
        StringBuffer sb = new StringBuffer(nums);
        for (int i = 0; i < n; i++) {
            if (Character.isLowerCase(sb.charAt(i))) {
                char temp = sb.charAt(j);
                sb.setCharAt(j, sb.charAt(i));
                sb.setCharAt(i, temp);
                j++;
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
            nums[begin] = nums[end];
            nums[end] = tmp;
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
                if (res > end - begin) {
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

            if (rowBegin <= rowEnd) {
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
        if (board == null || board.length == 0 || board[0].length == 0 || word.isEmpty())
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
        if (board == null || board.length == 0 || board[0].length == 0 || word.isEmpty())
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
        List<Integer> minlen = new ArrayList<Integer>(triangle.get(n - 1));
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
            if (nums[mid + 1] > nums[mid])
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
            if (nums[mid] > nums[end])//really goode question,mid can equal begin, but can only reach end-1;
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






}
