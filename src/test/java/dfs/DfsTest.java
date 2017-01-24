package dfs;

/**
 * Created by Tao on 1/23/2017.
 */

import org.junit.Before;
import org.junit.Test;

import java.beans.Transient;
import java.util.*;

import static org.junit.Assert.*;

public class DfsTest {
    DfsQuestion dq = null;

    @Before
    public void setup() {
        dq = new DfsQuestion();
    }

    @Test
    public void testNumberofIsalnds() {
        char[][] board = {{'1', '0', '1', '1', '1'}, {'1', '0', '1', '0', '1'}, {'1', '1', '1', '0', '1'}};
        // char [][]board={{'1'},{'1'}};
        System.out.println(dq.numIslandsByUnionFind(board));
    }
}
