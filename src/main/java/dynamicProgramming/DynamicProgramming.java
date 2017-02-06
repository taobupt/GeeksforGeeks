package dynamicProgramming;

/**
 * Created by Tao on 2/4/2017.
 */
public class DynamicProgramming {


    //bomb enemy
    public int maxKilledEnemies(char[][] grid) {
        if (grid.length == 0 || grid[0].length == 0)
            return 0;
        int m = grid.length, n = grid[0].length;
        int rowhits = 0;
        int[] colhits = new int[n];
        int res = 0;
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (grid[i][j] == 'W')
                    continue;

                //need to update
                if (j == 0 || grid[i][j - 1] == 'W') {
                    rowhits = 0;
                    for (int k = j; k < n && grid[i][k] != 'W'; ++k)
                        rowhits += grid[i][k] == 'E' ? 1 : 0;

                }
                //need to update colhits
                if (i == 0 || grid[i - 1][j] == 'W') {
                    colhits[j] = 0;
                    for (int k = i; k < m && grid[k][j] != 'W'; ++k)
                        colhits[j] += grid[k][j] == 'E' ? 1 : 0;
                }
                if (grid[i][j] == '0')
                    res = Math.max(res, colhits[j] + rowhits);
            }
        }
        return res;
    }
}
