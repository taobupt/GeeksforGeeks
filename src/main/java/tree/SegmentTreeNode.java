package tree;

/**
 * Created by tao on 2/9/17.
 */
public class SegmentTreeNode {
    public int start;
    public int end;
    public SegmentTreeNode left;
    public SegmentTreeNode right;
    public int maxValue;

    public SegmentTreeNode(int start, int end) {
        this.start = start;
        this.end = end;
        this.left = null;
        this.right = null;
    }

    public SegmentTreeNode(int start, int end, int maxValue) {
        this.start = start;
        this.end = end;
        this.left = null;
        this.right = null;
        this.maxValue = maxValue;
    }


}
