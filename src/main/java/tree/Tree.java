package tree;


import java.util.*;

/**
 * Created by Tao on 12/31/2016.
 */

class TreeNode{
    int val;
    TreeNode left;
    TreeNode right;
    TreeNode(int x){
        this.val=x;
        left=null;
        right=null;
    }
}
public class Tree {

    public Tree(){

    }

    //104 maximum depth of binary tree

    //recursive  way
    //time complexity is O(n),since it traverse the node once
    public int maxDepth(TreeNode root) {
        if (root == null)
            return 0;
        else
            return Math.max(maxDepth(root.left), maxDepth(root.right)) + 1;
    }

    //its corresponding stack version
    public int maxDepthStack(TreeNode root) {
        if (root == null)
            return 0;
        Stack<TreeNode> stk = new Stack<TreeNode>();
        Stack<Integer> value = new Stack<Integer>();
        stk.push(root);
        value.push(1);
        int max = 0;
        while (!stk.isEmpty()) {
            TreeNode node = stk.pop();
            int tmp = value.pop();
            max = Math.max(max, tmp);
            if (node.left != null) {
                stk.push(node.left);
                value.push(tmp + 1);
            }
            if (node.right != null) {
                stk.push(node.right);
                value.push(tmp + 1);
            }
        }
        return max;
    }


    //you can also use queue
    //iterative way

    public int maxDepthByQueue(TreeNode root) {
        int count = 0;
        Queue<TreeNode> dq = new LinkedList<TreeNode>();
        if (root == null)
            return 0;
        dq.offer(root);
        //push: add element in the front, useless
        while (!dq.isEmpty()) {
            int size = dq.size();
            while (size-- > 0) {
                TreeNode node = dq.poll();
                if (node.left != null)
                    dq.offer(node.left);
                if (node.right != null)
                    dq.offer(node.right);
            }
            count++;
        }
        return count;

    }


    //111 minimum depth of Binary Tree
    //recursive way
    //time complexity is O(n),since it traverse the node only once

    public int minDepth(TreeNode root) {
        if (root == null)
            return 0;
        if (root.left == null)
            return minDepth(root.right) + 1;
        else if (root.right == null)
            return minDepth(root.left) + 1;
        else
            return Math.min(minDepth(root.left), minDepth(root.right)) + 1;
    }

    //bfs way level order
    //time complexity is O(n),space complexity is O(n)
    public int minDepthByQueue(TreeNode root) {
        if (root == null)
            return 0;
        Queue<TreeNode> q = new LinkedList<TreeNode>();
        q.offer(root);
        int count = 0;
        while (!q.isEmpty()) {
            int size = q.size();
            count++;
            while (size-- > 0) {
                TreeNode node = q.poll();
                if (node.left == null && node.right == null)
                    return count;
                if (node.left != null)
                    q.offer(node.left);
                if (node.right != null)
                    q.offer(node.right);
            }
        }
        return count;
    }


    //235 lowest common ancestor of a binary search tree
    //recursive way
    //time complexity is O(h), requires O(H) extra space in function call stack for recursive function calls
    //we can avoid this by iterative solution
    public TreeNode lowestCommonAncestor(TreeNode root, TreeNode p, TreeNode q) {
        if (root == null || p == null || q == null)
            return null;
        if (root.val > Math.max(p.val, q.val))
            return lowestCommonAncestor(root.left, p, q);
        else if (root.val < Math.min(p.val, q.val))
            return lowestCommonAncestor(root.right, p, q);
        else
            return root;
    }

    //iterative way
    public TreeNode lowestCommonAncestorIterative(TreeNode root, TreeNode p, TreeNode q) {
        if (root == null || q == null || p == null)
            return null;
        TreeNode node = root;
        while (node != null) {
            if (node.val > Math.max(p.val, q.val))
                node = node.left;
            else if (node.val < Math.min(p.val, q.val))
                node = node.right;
            else
                return node;
        }
        return node;
    }


    // lowestCommonAncestor of binary tree
    public TreeNode lowestCommonAncestorBinaryTree(TreeNode root, TreeNode p, TreeNode q) {
        if (root == null || p == root || q == root)
            return root;
        TreeNode left = lowestCommonAncestorBinaryTree(root.left, p, q);
        TreeNode right = lowestCommonAncestorBinaryTree(root.right, p, q);
        if (left != null && right != null)
            return root;
        return left != null ? left : right;
    }

    //you can use stack, which is the best way to understand the nature of the problem
    //use stack to store store the path

    public List<TreeNode> stkToList(Stack<TreeNode> stk) {
        List<TreeNode> list = new ArrayList<TreeNode>(stk);//new Arraylist(Collection);
        return list;
    }

    public TreeNode lowestCommonAncestorBinaryTreeIterative(TreeNode root, TreeNode p, TreeNode q) {
        List<TreeNode> vp = new ArrayList<TreeNode>();
        List<TreeNode> vq = new ArrayList<TreeNode>();
        Stack<TreeNode> stk = new Stack<TreeNode>();
        //you should mark which nodes has been visited
        Set<TreeNode> set = new HashSet<TreeNode>();
        if (root == null || p == root || q == root)
            return root;
        stk.push(root);
        set.add(root);
        while (!stk.isEmpty()) {
            TreeNode node = stk.peek();
            if (node.left != null && !set.contains(node.left)) {
                stk.push(node.left);
                set.add(node.left);
                if (node.left == p) {
                    vp = stkToList(stk);
                    if (!vq.isEmpty())
                        break;

                }
                if (node.left == q) {
                    vq = stkToList(stk);
                    if (!vp.isEmpty())
                        break;
                }
                continue;
            }
            if (node.right != null && !set.contains(node.right)) {
                stk.push(node.right);
                set.add(node.right);
                if (node.right == p) {
                    vp = stkToList(stk);
                    if (!vq.isEmpty())
                        break;

                }
                if (node.right == q) {
                    vq = stkToList(stk);
                    if (!vp.isEmpty())
                        break;
                }
                continue;
            }
            stk.pop();
        }
        int index = Math.min(vp.size(), vq.size());
        int i = 0;
        for (; i < index; ++i) {
            if (vp.get(i) != vq.get(i))
                break;
        }
        if (i == 0 || vp.get(i - 1) == null)
            return null;
        return vp.get(i - 1);

    }











    //you can solve this by previous node
    //99 recover binary serach tree
    TreeNode pre11=null;
    TreeNode first=null;
    TreeNode second=null;
    public void inorderTree(TreeNode root){
        if(root==null)
            return;
        inorderTree(root.left);
        if(pre11!=null){
            if(pre11.val>root.val){
                if(first==null)
                    first=pre11;
                second=root;// not else, if there is only one chance that you will get into this case, if you use else, then second would be null, so you can't swap value later
            }
        }
        pre11=root;
        inorderTree(root.right);
    }

    public void recoverTree(TreeNode root){
        if(root==null)
            return;
        inorderTree(root);
        int tmp=first.val;
        first.val=second.val;
        second.val=tmp;
    }

    // actually you can slove this by morris traversal, which is the most optimal way
    //you just replace sout node.val with some code
    public void recoverTreeByMorrisTraversal(TreeNode root) {
        TreeNode pre = null;
        TreeNode cur = root;
        TreeNode first = null;
        TreeNode second = null;
        while (cur != null) {
            if (cur.left == null) {
                //replace sout value to some code;
                if (pre != null && pre.val > cur.val) {
                    if (first == null) {
                        first = pre;
                    }
                    second = cur;
                }
                pre = cur;
                cur = cur.right;
            } else {
                //find the precedessor
                TreeNode tmp = cur.left;
                while (tmp.right != null && tmp.right != cur) {
                    tmp = tmp.right;
                }
                if (tmp.right == null) {
                    tmp.right = cur;
                    cur = cur.left;
                } else {
                    tmp.right = null;
                    //this condition is very important
                    if (pre != null && pre.val > cur.val) {
                        if (first == null) {
                            first = pre;
                        }
                        second = cur;
                    }
                    pre = cur;
                    cur = cur.right;
                }
            }

        }
        if (first != null && second != null) {
            int val = first.val;
            first.val = second.val;
            second.val = val;
        }
    }





    //three ways to traverse preorder,inorder,postorder
    public void preorder(TreeNode root){
        if(root==null)
            return;
        System.out.println(root.val);
        preorder(root.left);
        preorder(root.right);
    }

    public void inorder(TreeNode root){
        if(root==null)
            return;
        inorder(root.left);
        System.out.println(root.val);
        inorder(root.right);
    }

    public void postorder(TreeNode root){
        if(root==null)
            return;
        postorder(root.left);
        postorder(root.right);
        System.out.println(root.val);
    }

    //stack way

    public void preorderStack(TreeNode root){
        if(root==null)
            return;
        Stack<TreeNode>stack=new Stack<TreeNode>();
        stack.push(root);
        TreeNode node=null;
        while(!stack.isEmpty()){
            node=stack.pop();
            System.out.println(node.val);
            if(node.right!=null)
                stack.push(node.right);
            if(node.left!=null)
                stack.push(node.left);
        }
    }

    public void inorderStack(TreeNode root){
        if(root==null)
            return;
        Stack<TreeNode>stk=new Stack<TreeNode>();
        TreeNode node=root;
        while(node!=null){
            stk.push(node);
            node=node.left;
        }

        while(!stk.isEmpty()){
            node=stk.pop();
            System.out.println(node.val);
            if(node.right!=null){
                node=node.right;
                while(node!=null){
                    stk.push(node);
                    node=node.left;
                }
            }
        }
    }

    public void postorderStack(TreeNode root){
        TreeNode pre=null;
        TreeNode cur=root;
        Stack<TreeNode>stk=new Stack<TreeNode>();
        stk.push(cur);
        while(!stk.isEmpty()){
            cur=stk.peek();
            if(pre==null||pre.left==cur||pre.right==cur){
                if(cur.left!=null)
                    stk.push(cur.left);
                else if(cur.right!=null)
                    stk.push(cur.right);
                else{
                    stk.pop();
                    System.out.println(cur.val);
                }
            }else if(cur.left==pre){
                if(cur.right!=null){
                    stk.push(cur.right);
                }else{
                    stk.pop();
                    System.out.println(cur.val);
                }
            }else if(cur.right==pre){
                stk.pop();
                System.out.println(cur.val);
            }
            pre=cur;
        }
    }


    //morris traversal
    public void inorderMorrisTraversal(TreeNode root){
        TreeNode cur=root;
        TreeNode prev=null;
        while(cur!=null){
            if(cur.left==null){
                System.out.println(cur.val);
                cur=cur.right;
            }else{
                //find predecessor
                prev=cur.left;
                while(prev.right!=null && prev.right!=cur){
                    prev=prev.right;
                }
                if(prev.right==null){
                    prev.right=cur;
                    cur=cur.left;
                }else{
                    prev.right=null;
                    System.out.println(cur.val);
                    cur=cur.right;
                }
            }
        }
    }

    //morris traversal
    public void preorderMorrisTraversal(TreeNode root){
        TreeNode cur=root;
        TreeNode prev=null;
        while(cur!=null){
            if(cur.left==null){
                System.out.println(cur.val);
                cur=cur.right;
            }else{
                //find predecessor
                prev=cur.left;
                while(prev.right!=null && prev.right!=cur){
                    prev=prev.right;
                }
                if(prev.right==null){
                    prev.right=cur;
                    System.out.println(cur.val);
                    cur=cur.left;
                }else{
                    prev.right=null;
                    cur=cur.right;
                }
            }
        }
    }

    public void reverse(TreeNode from,TreeNode to){
        if(from==to)
            return;
        TreeNode x=from;
        TreeNode y=to;
        TreeNode z=null;
        while(true){
            z=y.right;
            y.right=x;
            x=y;
            y=z;
            if(x==to)
                break;
        }
    }
    //something like the print the linkedlist reversely
    public void printReverse(TreeNode from,TreeNode to){
        //you can reverse two times, just like print list reverse,
        //you can also use recursive
        reverse(from,to);
        TreeNode p=to;
        while(true){
            System.out.println(p.val);
            if(p==from)
                break;
            p=p.right;
        }
        reverse(to,from);
    }

    //here is the recursive way to print linkedlist reversely

    public void printReverseRecursive(TreeNode from,TreeNode to){
        if(from==to){
            System.out.println(to.val);
            return;
        }
        printReverseRecursive(from.right,to);
        System.out.println(from.val);
    }
    public void postorderMorrisTraversal(TreeNode root){
        TreeNode dummy=new TreeNode(0);
        dummy.left=root;
        TreeNode cur=dummy;
        TreeNode prev=null;
        while(cur!=null){
            if(cur.left==null){
                cur=cur.right;
            }else{
                prev=cur.left;
                while(prev.right!=null && prev.right!=cur){
                    prev=prev.right;
                }
                if(prev.right==null){
                    prev.right=cur;
                    cur=cur.left;
                }else{
                    //printReverse(cur.left,prev);
                    printReverseRecursive(cur.left,prev);
                    prev.right=null;
                    cur=cur.right;
                }
            }
        }

    }

}
