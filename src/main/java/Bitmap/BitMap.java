package Bitmap;

/**
 * Created by tao on 1/24/17.
 */
public class BitMap {

    //461 Hamming Distance
    public int hammingDistance(int x, int y) {
        return Integer.bitCount(x ^ y);
    }

    //another version
    public int hammingDistanceBit(int x, int y) {
        int xor = x ^ y;
        int cnt = 0;
        for (int i = 0; i < 32; ++i) {
            cnt += (xor >> i) & 1;
        }
        return cnt;
    }

    //477 477. Total Hamming Distance
    public int totalHammingDistance(int[] nums) {
        int total = 0, n = nums.length;
        for (int i = 0; i < 32; ++i) {
            int cnt = 0;
            for (int j = 0; j < n; ++j) {
                cnt += (nums[j] >> i) & 1;
            }
            total += cnt * (n - cnt);
        }
        return total;
    }
}
