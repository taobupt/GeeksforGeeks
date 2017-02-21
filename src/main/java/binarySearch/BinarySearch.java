package binarySearch;

/**
 * Created by Tao on 2/21/2017.
 */
public class BinarySearch {
    public boolean doable(int[] L, int cuts, long maxLength) {
        if (maxLength == 0)
            return false;
        for (int x : L) {
            cuts -= x / maxLength;
            if (cuts <= 0)
                return true;
        }
        return false;
    }

    public int woodCut(int[] L, int k) {
        // write your code here
        long left = 0, right = 0, mid = 0;
        for (int num : L) {
            right = Math.max(num, right);
        }
        left = right / k;
        while (left < right) {
            mid = left + (right - left) / 2;
            if (doable(L, k, mid))
                left = mid + 1;
            else
                right = mid;
        }
        if (left == 0)
            return 0;
        return (int) (doable(L, k, left) ? left : left - 1);

    }
}
