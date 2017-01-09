package array;

import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

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


}
