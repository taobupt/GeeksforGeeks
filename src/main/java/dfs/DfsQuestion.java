package dfs;

import java.util.ArrayList;
import java.util.List;

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


    public void union(int x, int y, int[] parent) {
        int xx = find(x, parent);
        int yy = find(y, parent);
        if (xx != yy)
            parent[xx] = yy;
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

    public int countComponents(int n, int[][] edges) {
        int[] parent = new int[n];
        for (int i = 0; i < n; ++i)
            parent[i] = i;
        int res = n;
        for (int i = 0; i < edges.length; ++i) {
            res++;
            int xx = find(edges[i][0], parent);
            int yy = find(edges[i][1], parent);
            if (xx != yy) {
                parent[xx] = yy;
                res--;
            }
        }
        return res;
    }


    //you can also solve this by union find
    //the idea comes from the observation that if a region is NOT captured, it is connected to the boundary, so if we
    //coonect all the 'O' nodes on the boundary to a dummy node, and then connect each 'O' to its neighbour 'O' nodes,
    // then we can tell directly whether a 'O' node is captrued by checking whether it is connect to the dummy node
    //130 solved by union find
    public void solveByUnionFind(char[][] board) {
        if (board.length == 0 || board[0].length == 0)
            return;
        int m = board.length, n = board[0].length;
        int[] parent = new int[m * n + 1];//parent[m*n] is the dummy node
        for (int i = 0; i <= m * n; ++i)
            parent[i] = i;
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if ((i == 0 || j == 0 || i == m - 1 || j == n - 1) && board[i][j] == 'O') {
                    union(i * n + j, m * n, parent);//connect to the dummy node
                } else if (board[i][j] == 'O') {
                    if (i > 0 && board[i - 1][j] == 'O')
                        union(i * n + j, i * n + j - n, parent);
                    if (j > 0 && board[i][j - 1] == 'O')
                        union(i * n + j, i * n + j - 1, parent);
                    if (i < m - 1 && board[i + 1][j] == 'O')
                        union(i * n + j, i * n + n + j, parent);
                    if (j < n - 1 && board[i][j + 1] == 'O')
                        union(i * n + j, i * n + j + 1, parent);
                }
            }
        }
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (find(i * n + j, parent) != find(m * n, parent))
                    board[i][j] = 'X';
            }
        }
    }


    public List<Integer> numIslands2(int m, int n, int[][] positions) {
        List<Integer> res = new ArrayList<>();
        if (m == 0 || n == 0 || positions.length == 0)
            return res;
        int[] parent = new int[m * n];
        for (int i = 0; i < m * n; ++i)
            parent[i] = i;
        int[] cnt = new int[1];
        boolean[][] vis = new boolean[m][n];
        for (int i = 0; i < positions.length; ++i) {
            cnt[0]++;
            vis[positions[i][0]][positions[i][1]] = true;
            int p = positions[i][0] * n + positions[i][1];
            int q = 0;
            if (positions[i][0] > 0 && vis[positions[i][0] - 1][positions[i][1]]) {
                q = p - n;
                union(p, q, parent, cnt);
            }
            if (positions[i][1] > 0 && vis[positions[i][0]][positions[i][1] - 1]) {
                q = p - 1;
                union(p, q, parent, cnt);
            }
            if (positions[i][0] < m - 1 && vis[positions[i][0] + 1][positions[i][1]]) {
                q = p + n;
                union(p, q, parent, cnt);
            }
            if (positions[i][1] < n - 1 && vis[positions[i][0]][positions[i][1] + 1]) {
                q = p + 1;
                union(p, q, parent, cnt);
            }
            res.add(cnt[0]);
        }
        return res;
    }

}
