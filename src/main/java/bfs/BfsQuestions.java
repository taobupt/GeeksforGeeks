package bfs;

import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;
import java.util.Queue;

import common.Point;
/**
 * Created by tao on 1/23/17.
 */

public class BfsQuestions {


    //317 shortest distance from all buildings

    public int shortestDistacne(int[][] grid) {
        if (grid.length == 0 || grid[0].length == 0)
            return 0;
        int m = grid.length, n = grid[0].length;
        int[][] distance = new int[m][n];
        int[][] reach = new int[m][n];
        int buildingNum = 0;
        final int[] shift = new int[]{0, 1, 0, -1, 0};
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (grid[i][j] == 1) {
                    buildingNum++;
                    Queue<int[]> q = new LinkedList<>();
                    q.offer(new int[]{i, j});
                    boolean[][] vis = new boolean[m][n];
                    int level = 1;//this is very tricky
                    while (!q.isEmpty()) {
                        int size = q.size();
                        while (size-- > 0) {
                            int[] curr = q.poll();
                            for (int k = 0; k < 4; ++k) {
                                int x = curr[0] + shift[k];
                                int y = curr[1] + shift[k + 1];
                                if (x < 0 || x >= m || y < 0 || y >= n || grid[x][y] != 0 || vis[x][y])
                                    continue;
                                distance[x][y] += level;//in case of add the value, it ensure you can add one every level. that exactly I want
                                reach[x][y]++;
                                vis[x][y] = true;
                                q.offer(new int[]{x, y});
                            }
                        }
                        level++;
                    }
                }
            }
        }
        int shortest = Integer.MAX_VALUE;
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (grid[i][j] == 0 && reach[i][j] == buildingNum)
                    shortest = Math.min(shortest, distance[i][j]);
            }
        }
        return shortest == Integer.MAX_VALUE ? -1 : shortest;
    }


    //130 surrounded regions
    public void solve(char[][] board) {
        if (board.length == 0 || board[0].length == 0)
            return;
        int m = board.length, n = board[0].length;
        int[] dx = {1, -1, 0, 0};
        int[] dy = {0, 0, 1, -1};
        boolean[][] vis = new boolean[m][n];
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (board[i][j] == 'O') {
                    Queue<int[]> q = new LinkedList<>();
                    Queue<int[]> q1 = new LinkedList<>();
                    q.offer(new int[]{i, j});
                    q1.offer(new int[]{i, j});
                    vis[i][j] = true;
                    boolean fill = true;
                    while (!q.isEmpty()) {
                        int[] curr = q.poll();
                        for (int k = 0; k < 4; ++k) {
                            int x = curr[0] + dx[k];
                            int y = curr[1] + dy[k];
                            if (x < 0 || x >= m || y < 0 || y >= n) {
                                fill = false;
                                continue;
                            }
                            if (!vis[x][y] && board[x][y] == 'O') {
                                vis[x][y] = true;
                                q.offer(new int[]{x, y});
                                q1.offer(new int[]{x, y});
                            }
                        }
                    }
                    if (fill) {
                        while (!q1.isEmpty()) {
                            int[] curr = q1.poll();
                            board[curr[0]][curr[1]] = 'X';
                        }
                    }
                }
            }
        }
    }

    //an interesting solution
    //first, check the four border of the matrix, if there is a element is 'o', alter it and all its neightbor 'O' to '1'
    //then alter all the 'O' to 'X'
    //at last, alter all the '1' to 'O';

    public void check(char[][] board, int x, int y, int row, int col) {
        if (board[x][y] == 'O') {
            board[x][y] = '1';
            if (x > 1)
                check(board, x - 1, y, row, col);
            if (x < row - 1)
                check(board, x + 1, y, row, col);
            if (y > 1)
                check(board, x, y - 1, row, col);
            if (y < col - 1)
                check(board, x, y + 1, row, col);

        }
    }

    public void check(char[][] board, int x, int y, int row, int col, boolean[][] vis) {
        if (board[x][y] == 'O' && !vis[x][y]) {
            board[x][y] = '1';
            Queue<int[]> q = new LinkedList<>();
            int[] dx = {1, -1, 0, 0};
            int[] dy = {0, 0, 1, -1};
            vis[x][y] = true;
            q.offer(new int[]{x, y});
            while (!q.isEmpty()) {
                int[] curr = q.poll();
                for (int k = 0; k < 4; ++k) {
                    int nx = curr[0] + dx[k];//remember
                    int ny = curr[1] + dy[k];
                    if (nx < 0 || nx >= row || ny < 0 || ny >= col || vis[nx][ny] || board[nx][ny] != 'O')
                        continue;
                    vis[nx][ny] = true;
                    board[nx][ny] = '1';
                    q.offer(new int[]{nx, ny});
                }
            }
        }
    }

    public void solveConcise(char[][] board) {
        if (board.length == 0 || board[0].length == 0)
            return;
        int m = board.length, n = board[0].length;//remember
        boolean[][] vis = new boolean[m][n];
        for (int i = 0; i < m; ++i) {
            check(board, i, 0, m, n, vis);
            if (n > 1)
                check(board, i, n - 1, m, n, vis);
        }

        for (int i = 0; i < n; ++i) {
            check(board, 0, i, m, n, vis);
            if (m > 1)
                check(board, m - 1, i, m, n, vis);
        }

        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (board[i][j] == 'O')
                    board[i][j] = 'X';
                else if (board[i][j] == '1')
                    board[i][j] = 'O';
            }
        }
    }

    //TLE version
    public boolean check(int[][] matrix, int x, int y, int m, int n) {
        boolean canAtlantic = false;
        boolean canPacific = false;
        if (x == m - 1 && y == 0 || x == 0 && y == n - 1)
            return true;
        if (x == 0 || y == 0) {
            canPacific = true;
        }
        if (x == m - 1 || y == n - 1) {
            canAtlantic = true;
        }
        boolean[][] vis = new boolean[m][n];
        int[] dx = {1, -1, 0, 0};
        int[] dy = {0, 0, 1, -1};
        vis[x][y] = true;
        Queue<int[]> q = new LinkedList<>();
        q.offer(new int[]{x, y});
        while (!q.isEmpty()) {
            int[] cur = q.poll();
            for (int k = 0; k < 4; ++k) {
                int nx = cur[0] + dx[k];
                int ny = cur[1] + dy[k];
                if (nx < 0 || nx >= m || ny < 0 || ny >= n || vis[nx][ny] || matrix[nx][ny] > matrix[cur[0]][cur[1]])
                    continue;
                if (nx == m - 1 && ny == 0 || nx == 0 && ny == n - 1)
                    return true;
                if (nx == 0 || ny == 0) {
                    canPacific = true;
                    if (canAtlantic)
                        return true;
                }
                if (nx == m - 1 || ny == n - 1) {
                    canAtlantic = true;
                    if (canPacific)
                        return true;
                }
                vis[nx][ny] = true;
                q.offer(new int[]{nx, ny});

            }
        }
        return canAtlantic && canPacific;
    }

    // 417. Pacific Atlantic Water Flow
    public List<int[]> pacificAtlantic(int[][] matrix) {
        List<int[]> res = new ArrayList<>();
        if (matrix.length == 0 || matrix[0].length == 0)
            return res;
        int m = matrix.length, n = matrix[0].length;
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (check(matrix, i, j, m, n))
                    res.add(new int[]{i, j});
            }
        }
        return res;
    }

    public void check(int[][] matrix, Queue<int[]> q, boolean[][] vis) {
        int[] dx = {1, -1, 0, 0};
        int[] dy = {0, 0, 1, -1};
        int m = matrix.length, n = matrix[0].length;
        while (!q.isEmpty()) {
            int[] cur = q.poll();
            for (int k = 0; k < 4; ++k) {
                int nx = cur[0] + dx[k];
                int ny = cur[1] + dy[k];
                if (nx < 0 || nx >= m || ny < 0 || ny >= n || vis[nx][ny] || matrix[nx][ny] < matrix[cur[0]][cur[1]])
                    continue;
                vis[nx][ny] = true;
                q.offer(new int[]{nx, ny});

            }
        }
    }

    // public void check(int[][]matrix, Queue<int[]> queue, boolean[][]visited){
    //     int n = matrix.length, m = matrix[0].length;
    //     while(!queue.isEmpty()){
    //         int[] cur = queue.poll();
    //         for(int[] d:dir){
    //             int x = cur[0]+d[0];
    //             int y = cur[1]+d[1];
    //             if(x<0 || x>=n || y<0 || y>=m || visited[x][y] || matrix[x][y] < matrix[cur[0]][cur[1]]){
    //                 continue;
    //             }
    //             visited[x][y] = true;
    //             queue.offer(new int[]{x, y});
    //         }
    //     }
    // }
    // 417. Pacific Atlantic Water Flow
    public List<int[]> pacificAtlanticSaveTime(int[][] matrix) {
        List<int[]> res = new ArrayList<>();
        if (matrix.length == 0 || matrix[0].length == 0)
            return res;
        int m = matrix.length, n = matrix[0].length;
        boolean[][] pacificVisit = new boolean[m][n];
        boolean[][] atlanticVisit = new boolean[m][n];
        Queue<int[]> p = new LinkedList<>();
        Queue<int[]> q = new LinkedList<>();
        for (int i = 0; i < m; ++i) {
            p.offer(new int[]{i, 0});
            q.offer(new int[]{i, n - 1});
            pacificVisit[i][0] = true;
            atlanticVisit[i][n - 1] = true;
        }
        for (int i = 0; i < n; ++i) {
            p.offer(new int[]{0, i});
            q.offer(new int[]{m - 1, i});
            pacificVisit[0][i] = true;
            atlanticVisit[m - 1][i] = true;
        }
        check(matrix, p, pacificVisit);
        check(matrix, q, atlanticVisit);
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (pacificVisit[i][j] && atlanticVisit[i][j])
                    res.add(new int[]{i, j});
            }
        }
        return res;
    }


    //286 walls and gates
    public void wallsAndGates(int[][] rooms) {
        if (rooms.length == 0 || rooms[0].length == 0)
            return;
        int m = rooms.length, n = rooms[0].length;
        Queue<int[]> q = new LinkedList<>();
        for (int i = 0; i < m; ++i) {
            for (int j = 0; j < n; ++j) {
                if (rooms[i][j] == 0)
                    q.offer(new int[]{i, j});
            }
        }
        while (!q.isEmpty()) {
            int[] cur = q.poll();
            if (cur[0] > 0 && rooms[cur[0] - 1][cur[1]] == Integer.MAX_VALUE) {
                rooms[cur[0] - 1][cur[1]] = rooms[cur[0]][cur[1]] + 1;
                q.offer(new int[]{cur[0] - 1, cur[1]});
            }
            if (cur[1] > 0 && rooms[cur[0]][cur[1] - 1] == Integer.MAX_VALUE) {
                rooms[cur[0]][cur[1] - 1] = rooms[cur[0]][cur[1]] + 1;
                q.offer(new int[]{cur[0], cur[1] - 1});
            }
            if (cur[0] < m - 1 && rooms[cur[0] + 1][cur[1]] == Integer.MAX_VALUE) {
                rooms[cur[0] + 1][cur[1]] = rooms[cur[0]][cur[1]] + 1;
                q.offer(new int[]{cur[0] + 1, cur[1]});
            }
            if (cur[1] < n - 1 && rooms[cur[0]][cur[1] + 1] == Integer.MAX_VALUE) {
                rooms[cur[0]][cur[1] + 1] = rooms[cur[0]][cur[1]] + 1;
                q.offer(new int[]{cur[0], cur[1] + 1});
            }
        }

    }

    //490 the maze

    public boolean hasPath(int[][] maze, int[] start, int[] destination) {
        int m = maze.length, n = maze[0].length;
        if (start[0] == destination[0] && start[1] == destination[1]) return true;
        int[][] dir = new int[][]{{-1, 0}, {0, 1}, {1, 0}, {0, -1}};
        boolean[][] visited = new boolean[m][n];
        LinkedList<Point> list = new LinkedList<>();
        visited[start[0]][start[1]] = true;
        list.offer(new Point(start[0], start[1]));
        while (!list.isEmpty()) {
            Point p = list.poll();
            int x = p.x, y = p.y;
            for (int i = 0; i < 4; i++) {
                int xx = x, yy = y;
                while (xx >= 0 && xx < m && yy >= 0 && yy < n && maze[xx][yy] == 0) {
                    xx += dir[i][0];
                    yy += dir[i][1];
                }
                xx -= dir[i][0];
                yy -= dir[i][1];
                if (visited[xx][yy]) continue;
                visited[xx][yy] = true;
                if (xx == destination[0] && yy == destination[1]) return true;
                list.offer(new Point(xx, yy));
            }
        }
        return false;

    }
}
