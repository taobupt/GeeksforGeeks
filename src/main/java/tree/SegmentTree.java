package tree;

/**
 * Created by tao on 2/9/17.
 */

public class SegmentTree {
    //just an interval
    public SegmentTreeNode build(int start, int end) {
        if (start > end)//in case of invalid node
            return null;
        if (start == end)
            return new SegmentTreeNode(start, end);
        SegmentTreeNode root = new SegmentTreeNode(start, end);
        root.left = build(start, (end - start) / 2 + start);//in case of overflow
        root.right = build((end - start) / 2 + start + 1, end);//in case of overflow
        return root;
    }

    public SegmentTreeNode build(int A[], int start, int end) {
        if (start > end)
            return null;
        else if (start == end) {
            SegmentTreeNode root = new SegmentTreeNode(start, start, A[start]);
            return root;
        } else {
            SegmentTreeNode root = new SegmentTreeNode(start, end, 0);
            root.left = build(A, start, (end - start) / 2 + start);
            root.right = build(A, (end - start) / 2 + start + 1, end);
            root.maxValue = Math.max(root.left.maxValue, root.right.maxValue);
            return root;
        }

    }

    //an array, and its maxvalue
    public SegmentTreeNode build(int[] A) {
        return build(A, 0, A.length - 1);
    }

    //query maxvalue
    public int query(SegmentTreeNode root, int start, int end) {
        if (root.end < start || root.start > end)
            return -0x7fffffff;
        if (root.left == null && root.right == null)
            return root.maxValue;
        if (start >= root.right.start)
            return query(root.right, start, end);
        else if (end <= root.left.end)
            return query(root.left, start, end);
        else
            return Math.max(query(root.left, start, root.left.end), query(root.right, root.right.start, end));
    }

    //query count,treat the maximal value as the count
    public int queryCount(SegmentTreeNode root, int start, int end) {
        if (root == null || root.end < start || root.start > end)
            return 0;
        if (root.left == null && root.right == null)
            return root.maxValue;
        if (root.right.start <= start)
            return queryCount(root.right, start, end);
        else if (root.left.end >= end)
            return queryCount(root.left, start, end);
        return queryCount(root.left, start, root.left.end) + queryCount(root.right, root.right.start, end);
    }


    //modify
    public void modify(SegmentTreeNode root, int index, int value) {
        if (root == null)
            return;
        if (root.left == null && root.right == null) {
            if (root.start == index)
                root.maxValue = value;
            return;
        }
        modify(root.left, index, value);
        modify(root.right, index, value);
        root.maxValue = Math.max(root.left.maxValue, root.right.maxValue);
    }

}
