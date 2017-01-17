package hashTable;

import java.util.*;

/**
 * Created by taobupt on 1/17/2017.
 */
public class HASHTABLE {

    //477 number of boomeranges
    public int getDistance(int[] x1, int[] x2) {
        return (x1[0] - x2[0]) * (x1[0] - x2[0]) + (x1[1] - x2[1]) * (x1[1] - x2[1]);
    }

    public int numberOfBoomerangs(int[][] points) {
        List<Map<Integer, Integer>> li = new ArrayList<>();
        int n = points.length;
        for (int i = 0; i < n; ++i) {
            Map<Integer, Integer> map = new HashMap<>();
            for (int j = 0; j < n; ++j) {
                int dist = getDistance(points[i], points[j]);
                if (map.containsKey(dist)) {
                    map.put(dist, map.get(dist) + 1);
                } else
                    map.put(dist, 1);
            }
            li.add(map);
        }
        int cnt = 0;
        for (int i = 0; i < li.size(); ++i) {
            for (Map.Entry<Integer, Integer> entry : li.get(i).entrySet()) {
                int val = entry.getValue();
                if (val >= 2) {
                    cnt += (val - 1) * val;
                }
            }
        }
        return cnt;
    }

}
