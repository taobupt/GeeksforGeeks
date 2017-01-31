package greedy;

/**
 * Created by Tao on 1/30/2017.
 */

import org.junit.Before;
import org.junit.Test;

import java.beans.Transient;
import java.util.*;

import static org.junit.Assert.*;

public class TestGreedy {
    Greedy gd = null;

    @Before
    public void setup() {
        gd = new Greedy();
    }

    @Test
    public void testFindPermuation() {
        String s = "DDIIDI";
        int[] res = gd.findPermutation(s);
        for (int x : res)
            System.out.println(x);
    }

    @Test
    public void testfindMinArrowshots() {
        //int[][]points={{1,2},{1,3},{1,5},{2,5},{2,7},{1,10}};
        int[][] points = {{10, 16}, {2, 8}, {1, 6}, {7, 12}};
        System.out.println(gd.findMinArrowShots(points));
    }
}
