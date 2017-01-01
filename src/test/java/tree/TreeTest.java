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
        root = new TreeNode(10);
        root.left = new TreeNode(12);
        root.right = new TreeNode(15);
        root.left.left = new TreeNode(25);
        root.left.right = new TreeNode(30);
        root.right.left = new TreeNode(36);
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
}