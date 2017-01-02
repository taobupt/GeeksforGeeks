package tree;

import org.junit.Before;
import org.junit.Test;

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
        root = new TreeNode(3);
        root.left = new TreeNode(5);
        root.right = new TreeNode(1);
        root.left.left = new TreeNode(6);
        root.left.right = new TreeNode(2);
        root.right.left = new TreeNode(0);
        root.right.right = new TreeNode(8);
        root.left.right.left = new TreeNode(7);
        root.left.right.right = new TreeNode(4);
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
}