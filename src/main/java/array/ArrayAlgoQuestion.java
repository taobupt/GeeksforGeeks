package array;

import java.util.HashMap;
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

}
