package tree;

import org.junit.Before;
import org.junit.Test;

import java.util.ArrayList;
import java.util.List;

import static org.junit.Assert.*;

/**
 * Created by Tao on 12/31/2016.
 */
public class TreeTest {

    Tree t;
    TreeNode root;

    @Before
    public void setUp() throws Exception {
        t = new Tree();
        root = new TreeNode(5);
        root.left = new TreeNode(3);
        root.right = new TreeNode(7);
        root.left.left = new TreeNode(1);
        root.left.right = new TreeNode(4);
        root.right.left = new TreeNode(6);
        root.right.right = new TreeNode(8);
//        root.left.left.left = new TreeNode(7);
//        root.left.left.right = new TreeNode(2);
//        root.right.right.right=new TreeNode(1);
    }

    @Test
    public void testPreorder() throws Exception {
        t.preorder(root);
    }

    @Test
    public void testInorder() throws Exception {
        t.inorder(root);
    }

    @Test
    public void testPostorder() throws Exception {
        t.postorder(root);
    }

    @Test
    public void testPreorderStack() throws Exception {
        t.preorderStack(root);
    }

    @Test
    public void testInorderStack() throws Exception {
        t.inorderStack(root);
    }

    @Test
    public void testPostorderStack() throws Exception {
        t.postorderStack(root);
    }

    @Test
    public void testInorderMorrisTraversal() throws Exception {
        t.inorderMorrisTraversal(root);
    }

    @Test
    public void testPreorderMorrisTraversal() {
        t.preorderMorrisTraversal(root);
    }

    @Test
    public void testPostorderMorrisTraveral() {
        t.postorderMorrisTraversal(root);
    }

    @Test
    public void testLowestCommonAncestorBinaryTreeIterative() {
        TreeNode res = t.lowestCommonAncestorBinaryTreeIterative(root, root.left, root.left.right.right);
        assertEquals(res, root.left);
    }

    @Test
    public void testBinaryTreePaths() {
        List<String> res = t.binaryTreePaths(root);
        for (String str : res)
            System.out.println(str);
    }

    @Test
    public void testBinaryTreePathsIterative() {
        List<String> res = t.binaryTreePathsIterative(root);
        for (String str : res)
            System.out.println(str);
    }

    @Test
    public void testBinaryTreePathsIterativeByStack() {
        List<String> res = t.binaryTreePathsIterativeStack(root);
        for (String str : res)
            System.out.println(str);
    }

    @Test
    public void testBinaryTreePathsIterativeByQueue() {
        List<String> res = t.binaryTreePathsIterativeQueue(root);
        for (String str : res)
            System.out.println(str);

    }

    @Test
    public void testLevelOrder() {
        t.levelOrder(root);
    }

    @Test
    public void testLevelOrderRecursive() {
        List<List<Integer>> res = t.levelOrderRecursive(root);
        for (List<Integer> it : res) {
            for (Integer ite : it)
                System.out.print(ite + " ");
            System.out.println();
        }

    }

    @Test
    public void testIsSameTreeIterative() {
        t.isSameTreeIterative(root, root);
    }

    @Test
    public void testIsSymmetricIterative() {
        t.isSymmetricIterative(root);
    }

    @Test
    public void testHasPathSum() {
        t.hasPathSumIterative(root, 22);
    }

    @Test
    public void testPostorderByStack() {
        t.postorderByStack(root);
    }

    @Test
    public void testCountNode() {
        t.countNodesLastLevel(root);
    }

    @Test
    public void testZigzagOrder() {
        t.zigzagLevelOrder(root);
    }

    @Test
    public void testBuildTree() {
        int[] preorder = {1, 2, 4, 5, 3, 6, 7};
        int[] inorder = {4, 2, 5, 1, 6, 3, 7};
        t.bulidTreeIterative(preorder, inorder);
    }

    @Test
    public void testKthsmallest() {
        assertEquals(t.kthSmallestIterative(root, 6), 7);
    }

    @Test
    public void testFlatten() {
        //t.flattenReversePreorder(root);
        t.flattenByLeft(root);
        t.levelOrder(root);
    }

}