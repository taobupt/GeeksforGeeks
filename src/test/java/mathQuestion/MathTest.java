package mathQuestion;

/**
 * Created by taobupt on 1/17/2017.
 */


import org.junit.Before;
import org.junit.Test;

import java.util.*;

import static org.junit.Assert.*;

public class MathTest {
    MathQuestion mq = null;

    @Before
    public void setup() {
        mq = new MathQuestion();
    }

    @Test
    public void testRotateFunction() {
        int[] A = {-2147483648, -2147483648};
        System.out.println(mq.maxRotateFunction(A));
    }

    @Test
    public void testReflection() {
        int[][] points = {{1, 1}, {-1, 1}};
        mq.isReflected(points);
    }
}
