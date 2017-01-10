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












}
