package bfs;

/**
 * Created by tao on 1/23/17.
 */


import org.junit.Before;
import org.junit.Test;

import java.beans.Transient;
import java.util.*;

import static org.junit.Assert.*;

public class BfsTest {

    BfsQuestions bq = null;

    @Before
    public void setup() {
        bq = new BfsQuestions();
    }

    @Test
    public void testSurround() {
        //["XXXX","XOOX","XXOX","XOXX"]
        char[][] board = {{'O', 'O', 'O'}, {'O', 'O', 'O'}, {'O', 'O', 'O'}};
        bq.solveConcise(board);
    }

    @Test
    public void testPacific() {
        int[][] board = {{1, 2, 2, 3, 5}, {3, 2, 3, 4, 4}, {2, 4, 5, 3, 1}, {6, 7, 1, 4, 5}, {5, 1, 1, 2, 4}};
        bq.pacificAtlantic(board);
    }

    @Test
    public void testMaze() {
        int[][] maze = {{0, 0, 1, 0, 0}, {0, 0, 0, 0, 0}, {0, 0, 0, 1, 0}, {1, 1, 0, 1, 1}, {0, 0, 0, 0, 0}};
        int[] start = {0, 4};
        int[] des = {3, 2};
        System.out.println(bq.hasPath(maze, start, des));
    }

}
