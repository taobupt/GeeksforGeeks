package tree;

import common.Interval;

import java.util.ArrayList;

/**
 * Created by tao on 2/9/17.
 */

class Node {
    public int minValue;
    public int start;
    public int end;
    public Node left;
    public Node right;

    public Node(int start, int end, int minValue) {
        this.minValue = minValue;
        this.left = null;
        this.right = null;
        this.start = start;
        this.end = end;
    }
}

public class SegmentTreeQuestions {
    //lintcode interval minimum number

    public Node buildTree(int[] A, int start, int end) {
        if (start > end)
            return null;
        if (start == end)
            return new Node(start, end, A[end]);
        Node root = new Node(start, end, 0);
        root.left = buildTree(A, start, (end - start) / 2 + start);
        root.right = buildTree(A, (end - start) / 2 + start + 1, end);
        root.minValue = Math.min(root.left.minValue, root.right.minValue);
        return root;
    }

    //why this is faster
    public int query(Node root, int start, int end) {
        if (start > root.end || end < root.start)
            return Integer.MAX_VALUE;
        if (start <= root.start && end >= root.end)
            return root.minValue;
        int l = query(root.left, start, end);
        int r = query(root.right, start, end);
        return l < r ? l : r;
    }

    public ArrayList<Integer> intervalMinNumber(int[] A, ArrayList<Interval> queries) {
        ArrayList<Integer> res = new ArrayList<>();
        int n = A.length;
        Node root = buildTree(A, 0, n - 1);
        for (Interval interval : queries) {
            res.add(query(root, interval.start, interval.end));
        }
        return res;
    }

    //interval sum
}
