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
        Map<Double, ArrayList<Integer>> map = new HashMap<>();
        double potentialLine = 0;
        for (int i = 0; i < n; ++i) {
            minX = Math.min(minX, points[i][0]);
            maxX = Math.max(maxX, points[i][0]);
            if (map.containsKey(1.0 * points[i][0])) {
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
        for (int i = 0; i < n; ++i) {
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
        return 2 * leftcnt == (n - onTheLine);
    }

}
