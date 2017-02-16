package tree;

import java.util.ArrayList;

/**
 * Created by taobupt on 2/10/2017.
 */
class Node1 {
    public int start;
    public int end;
    public int count;
    public Node1 left;
    public Node1 right;

    public Node1(int start, int end, int count) {
        this.start = start;
        this.end = end;
        this.count = count;
    }
}

public class Solution {


    public Node1 buildTree(int start, int end) {
        if (start > end)
            return null;
        if (start == end)
            return new Node1(start, start, 0);
        Node1 root = new Node1(start, end, 0);
        root.left = buildTree(start, (end - start) / 2 + start);
        root.right = buildTree((end - start) / 2 + 1 + start, end);
        return root;
    }

    public void modify(Node1 Node1, int val) {
        if (Node1 == null || val > Node1.end || val < Node1.start)
            return;
        if (Node1.left == null && Node1.right == null) {
            if (Node1.start == val)
                Node1.count++;
            return;
        }
        if (val >= Node1.right.start)
            modify(Node1.right, val);
        if (val <= Node1.left.end)
            modify(Node1.left, val);
        Node1.count = Node1.left.count + Node1.right.count;
    }

    public int query(Node1 root, int val) {
        if (root == null)
            return 0;
        if (val > root.end)
            return root.count;
        if (val <= root.start)
            return 0;
        int l = query(root.left, val);
        int r = query(root.right, val);
        return l + r;
    }

    public ArrayList<Integer> countOfSmallerNumber(int[] A, int[] queries) {
        // write your code here
        Node1 root = buildTree(0, 10000);
        for (int x : A) {
            modify(root, x);
        }
        ArrayList<Integer> res = new ArrayList<Integer>();
        for (int q : queries) {
            res.add(query(root, q));
        }
        return res;


        //for smaller integer itself
        //ArrayList<Integer>res=new ArrayList<Integer>();
//        for(int x:A){
//            res.add(query(root,x));
//            modify(root,x);
//        }
//
//        return res;
    }
}