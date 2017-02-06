package graph;

/**
 * Created by Tao on 2/5/2017.
 */

import org.junit.Before;
import org.junit.Test;

import java.beans.Transient;
import java.util.*;

import static org.junit.Assert.*;

public class GraphTest {
    Graph gh = null;

    @Before
    public void setup() {
        gh = new Graph();
    }

    @Test
    public void testCalcEquation() {
        String[][] equations = {{"a", "b"}, {"b", "c"}};
        double[] values = {2.0, 3.0};
        String[][] queries = {{"a", "c"}, {"b", "c"}, {"a", "e"}, {"a", "a"}, {"x", "x"}};
        System.out.println(gh.calcEquation(equations, values, queries));
    }
}
