package dfs;

/**
 * Created by Tao on 1/23/2017.
 */


class UnionFind {
    int[] parnet = null;

    public UnionFind() {
        parnet = new int[1000000];
    }

    public int find(int x) {
        while (x != parnet[x]) {
            parnet[x] = parnet[parnet[x]];
            x = parnet[x];
        }
        return x;
    }

    public boolean mix(int x, int y) {
        int xx = find(x);
        int yy = find(y);
        if (xx != yy) {
            parnet[xx] = yy;
            return true;
        }
        return false;
    }
}

public class DfsQuestion {


    //200. Number of Islands
    //dfs way
    public void dfs(char[][] grid, int x, int y, boolean[][] vis) {
        if (x < 0 || x >= grid.length || y < 0 || y >= grid[0].length || vis[x][y] || grid[x][y] == '0')
            return;
        vis[x][y] = true;
        dfs(grid, x + 1, y, vis);
        dfs(grid, x - 1, y, vis);
        dfs(grid, x, y + 1, vis);
        dfs(grid, x, y - 1, vis);
    }

    public int numIslands(char[][] grid) {
        if (grid.length == 0 || grid[0].length == 0)
            return 0;
        int m = grid.length, n = grid[0].length;
        int res = 0;
        boolean[][] visit = new boolean[m][n];
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (!visit[i][j] && grid[i][j] == '1') {
                    dfs(grid, i, j, visit);
                    res++;
                }
            }
        }
        return res;
    }

    //you can also use union find
    public int find(int x, int[] parent) {
        while (x != parent[x]) {
            parent[x] = parent[parent[x]];
            x = parent[x];
        }
        return x;
    }

    public void union(int x, int y, int[] parent, int[] cnt) {
        int xx = find(x, parent);
        int yy = find(y, parent);
        if (xx == yy)
            return;
        parent[xx] = yy;
        cnt[0]--;
    }

    public int numIslandsByUnionFind(char[][] grid) {
        if (grid.length == 0 || grid[0].length == 0)
            return 0;
        int m = grid.length, n = grid[0].length;
        int[] cnt = new int[1];
        cnt[0] = 0;
        int[] parent = new int[m * n];
        for (int i = 0; i < m * n; ++i)
            parent[i] = i;
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (grid[i][j] == '1') {
                    cnt[0]++;
                    int p = i * n + j;
                    int q = 0;
                    if (i > 0 && grid[i - 1][j] == '1') {   //not else if you can remember
                        q = p - n;
                        union(p, q, parent, cnt);
                    }
                    if (j > 0 && grid[i][j - 1] == '1') {
                        q = p - 1;
                        union(p, q, parent, cnt);
                    }
                    if (i < m - 1 && grid[i + 1][j] == '1') {
                        q = p + n;
                        union(p, q, parent, cnt);
                    }
                    if (j < n - 1 && grid[i][j + 1] == '1') {
                        q = p + 1;
                        union(p, q, parent, cnt);
                    }
                }
            }
        }
        return cnt[0];
    }
}
