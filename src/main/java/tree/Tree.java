package tree;


import java.util.*;

/**
 * Created by Tao on 12/31/2016.
 */


//dque is much faster
class Codec {

    // Encodes a tree to a single string.
    public String serialize(TreeNode root) {
        StringBuffer sb = new StringBuffer();
        serializeR(root, sb);
        return sb.toString();
    }

    // Decodes your encoded data to tree.
    public TreeNode deserialize(String data) {
        Deque<String> dq = new LinkedList<String>();
        dq.addAll(Arrays.asList(data.split(" ")));
        return deserialize(dq);
    }

    private TreeNode deserialize(Deque<String> datas) {
        String val = datas.remove();
        if (val.equals("@"))
            return null;
        TreeNode root = new TreeNode(Integer.valueOf(val));
        root.left = deserialize(datas);
        root.right = deserialize(datas);
        return root;
    }

    private void serializeR(TreeNode root, StringBuffer res) {
        if (root != null) {
            res.append(root.val).append(" ");
            serializeR(root.left, res);
            serializeR(root.right, res);
        } else {
            res.append("@").append(" ");
        }
    }
}




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

class TreeLinkNode {
    int val;
    TreeLinkNode left, right, next;

    TreeLinkNode(int x) {
        this.val = x;
        this.left = null;
        this.right = null;
        this.next = null;
    }
}
class info {
    int size;
    boolean isBST;
    int minValue;
    int maxValue;

    public info() {
        this.size = 0;
        this.isBST = true;
        this.minValue = Integer.MAX_VALUE;
        this.maxValue = Integer.MIN_VALUE;
    }
}


class uniInfo {
    int size;
    boolean isUni;
    int value;

    public uniInfo() {
        this.size = 0;
        this.isUni = true;
        this.value = Integer.MIN_VALUE;
    }
}


class MyNode {
    TreeNode node;
    int lb;
    int rb;
    int index;

    public MyNode(TreeNode n, int theLeft, int theRight) {
        this.node = n;
        this.lb = theLeft;
        this.rb = theRight;
    }
}

class Tuple<X, Y> {
    public final X x;
    public final Y y;

    public Tuple(X x, Y y) {
        this.x = x;
        this.y = y;
    }
}

class BalancedResult {
    boolean isBalanced;
    int size;

    BalancedResult(boolean b, int s) {
        isBalanced = b;
        size = s;
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

    public int minDepthByStack(TreeNode root) {
        if (root == null)
            return 0;
        Stack<Tuple<TreeNode, Integer>> stk = new Stack<>();
        stk.push(new Tuple(root, 1));
        int res = Integer.MAX_VALUE;
        while (!stk.isEmpty()) {
            Tuple t = stk.pop();
            TreeNode node = (TreeNode) t.x;
            int val = ((Integer) t.y).intValue();
            if (node.left == null && node.right == null) {
                res = Math.min(res, val);
            }
            if (node.left != null) {
                stk.push(new Tuple(node.left, val + 1));
            }
            if (node.right != null) {
                stk.push(new Tuple(node.right, val + 1));
            }
        }
        return res;
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


    //257 binary tree paths
    //recursive way
    //leafpath
    //leaf to path

    public void dfsBinaryTreePaths(TreeNode node, String path, List<String> res) {
        if (node == null)
            return;
        if (node.left == null && node.right == null) {
            res.add(path + "" + node.val);
            return;
        }
        if (node.left != null)
            dfsBinaryTreePaths(node.left, path + node.val + "->", res);
        if (node.right != null)
            dfsBinaryTreePaths(node.right, path + node.val + "->", res);
    }

    public List<String> binaryTreePaths(TreeNode root) {
        List<String> res = new ArrayList<String>();
        if (root == null)
            return res;
        dfsBinaryTreePaths(root, "", res);
        return res;
    }

    //actually you can use a stack
    //these in dfs order, but queue in level order print binary treePaths
    public List<String> binaryTreePathsIterativeStack(TreeNode root) {
        Stack<Tuple<TreeNode, StringBuffer>> stk = new Stack<Tuple<TreeNode, StringBuffer>>();
        List<String> res = new ArrayList<String>();
        if (root == null)
            return res;
        stk.push(new Tuple(root, new StringBuffer("")));
        while (!stk.isEmpty()) {
            Tuple t = stk.pop();
            TreeNode node = (TreeNode) t.x;
            StringBuffer path = new StringBuffer();
            path.append(((StringBuffer) t.y).toString() + "->" + node.val);
            if (node.left == null && node.right == null) {
                res.add(path.toString().substring(2));
            }
            if (node.right != null) {
                stk.push(new Tuple(node.right, path));
            }
            if (node.left != null) {
                stk.push(new Tuple(node.left, path));
            }
        }
        return res;
    }

    // the same as stack, but different order
    //these leaves in level order;but stack in dfs order
    public List<String> binaryTreePathsIterativeQueue(TreeNode root) {
        Queue<Tuple<TreeNode, StringBuffer>> q = new LinkedList<Tuple<TreeNode, StringBuffer>>();
        List<String> res = new ArrayList<String>();
        if (root == null)
            return res;
        q.offer(new Tuple(root, new StringBuffer("")));
        while (!q.isEmpty()) {
            Tuple t = q.poll();
            TreeNode node = (TreeNode) t.x;
            StringBuffer path = new StringBuffer();
            path.append(((StringBuffer) t.y).toString() + "->" + node.val);
            if (node.left == null && node.right == null) {
                res.add(path.toString().substring(2));
            }
            if (node.left != null) {
                q.offer(new Tuple(node.left, path));
            }
            if (node.right != null) {
                q.offer(new Tuple(node.right, path));
            }
        }
        return res;
    }

    //iterative way
    //use an hashmap to store the parent of the node
    public List<String> binaryTreePathsIterative(TreeNode root) {
        Map<TreeNode, TreeNode> parent = new HashMap<TreeNode, TreeNode>();
        List<String> res = new ArrayList<String>();
        if (root == null)
            return res;
        Queue<TreeNode> q = new LinkedList<TreeNode>();
        q.offer(root);
        while (!q.isEmpty()) {
            TreeNode node = q.poll();
            if (node.left == null && node.right == null) {
                //this is a leaf node
                //find its parent one by one
                TreeNode tmp = node;
                String str = "" + tmp.val;
                while (parent.containsKey(tmp)) {
                    str = parent.get(tmp).val + "->" + str;
                    tmp = parent.get(tmp);
                }
                res.add(str);
            }
            if (node.left != null) {
                parent.put(node.left, node);
                q.offer(node.left);
            }
            if (node.right != null) {
                parent.put(node.right, node);
                q.offer(node.right);
            }
        }
        return res;
    }

    //leetcode 226 invert binary tree

    public TreeNode invertTree(TreeNode root) {
        if (root == null)
            return null;
        TreeNode saveLeft = root.left;
        root.left = invertTree(root.right);
        root.right = invertTree(saveLeft);
        return root;
    }

    //follow up iterative way
    public TreeNode invertTreeIterative(TreeNode root) {
        if (root == null)
            return null;
        Stack<TreeNode> stk = new Stack<TreeNode>();
        stk.push(root);
        while (!stk.isEmpty()) {
            TreeNode node = stk.pop();
            TreeNode tmp = node.left;
            node.left = node.right;
            node.right = tmp;
            if (node.left != null) {
                stk.push(node.left);
            }
            if (node.right != null) {
                stk.push(node.right);
            }
        }
        return root;
    }

    //you can also use a queue
    //just traversal all the nodes
    //swap their left child and right child
    public TreeNode invertTreeByQueue(TreeNode root) {
        if (root == null)
            return null;
        Queue<TreeNode> q = new LinkedList<TreeNode>();
        q.offer(root);
        while (!q.isEmpty()) {
            TreeNode node = q.poll();
            TreeNode tmp = node.left;
            node.left = node.right;
            node.right = tmp;
            if (node.left != null) {
                q.offer(node.left);
            }
            if (node.right != null) {
                q.offer(node.right);
            }
        }
        return root;
    }


    //leetcode 100 same tree
    //recursive way
    public boolean isSameTree(TreeNode p, TreeNode q) {
        if (p == null || q == null)
            return p == q;
        return (p.val == q.val) && isSameTree(p.left, q.left) && isSameTree(p.right, q.right);
    }

    //iterative way
    //p.left!=q.left is forbidden, you canot compare the address
    public boolean isSameTreeIterative(TreeNode p, TreeNode q) {
        if (p == null || q == null)
            return p == q;
        Stack<TreeNode> pstk = new Stack<TreeNode>();
        Stack<TreeNode> qstk = new Stack<TreeNode>();
        pstk.push(p);
        qstk.push(q);
        while (!pstk.isEmpty() && !qstk.isEmpty()) {
            TreeNode pnode = pstk.pop();
            TreeNode qnode = qstk.pop();
            if (pnode.val != qnode.val)
                return false;
            if (pnode.left != null)
                pstk.push(pnode.left);
            if (qnode.left != null)
                qstk.push(qnode.left);
            if (pstk.size() != qstk.size())
                return false;
            if (pnode.right != null)
                pstk.push(pnode.right);
            if (qnode.right != null)
                qstk.push(qnode.right);
            if (pstk.size() != qstk.size())
                return false;

        }
        return pstk.size() == qstk.size();
    }

    //110 balanced binary tree
    //bottom up approach
    //time complexity O(N)
    //each node in the tree only need to be accessed once, thus the time complexity is O(N)
    public int isBalancedDfs(TreeNode root) {
        if (root == null)
            return 0;
        int left = isBalancedDfs(root.left);
        int right = isBalancedDfs(root.right);
        if (left < 0 || right < 0 || Math.abs(left - right) > 1)
            return -1;
        return Math.max(left, right) + 1;
    }

    public boolean isBalanced(TreeNode root) {
        if (root == null)
            return true;
        return isBalancedDfs(root) != -1;
    }

    //o(n2) solution
    //this method checks whether the tree is balanced strictly according to the defination of balanced binary tree
    //for the current node root,calling depth() for its depth and right children acataully has to access all of its children
    //
    public boolean isBalancedON2(TreeNode node) {
        if (node == null)
            return true;
        int left = maxDepth(node.left);
        int right = maxDepth(node.right);
        return Math.abs(left - right) <= 1 && isBalancedON2(node.left) && isBalancedON2(node.right);
    }

    //third way

    private BalancedResult isBalanced_res(TreeNode node) {
        if (node == null)
            return new BalancedResult(true, 0);
        BalancedResult left = isBalanced_res(node.left);
        BalancedResult right = isBalanced_res(node.right);
        if (!left.isBalanced || !right.isBalanced) {
            return new BalancedResult(false, -1);
        } else {
            if (Math.abs(left.size - right.size) > 1) {
                return new BalancedResult(false, -1);
            } else {
                int newSize = Math.max(left.size, right.size) + 1;
                return new BalancedResult(true, newSize);
            }
        }
    }

    public boolean isBalancedByResult(TreeNode node) {
        return isBalanced_res(node).isBalanced;
    }

    //one way is to pack them together into a class and return object of the class
    // another way is to use instance variable
    //third way is to reuse the depth information, since depth of tree can't be negative, we could
    //use depth=-1 to flag the tree as not balanced,


    //101 symmetric tree
    //recursive way
    public boolean isSymmetric(TreeNode root) {
        if (root == null)
            return true;
        return isSymmetricHelper(root.left, root.right);
    }

    public boolean isSymmetricHelper(TreeNode p, TreeNode q) {
        if (p == null || q == null)
            return p == q;
        return p.val == q.val && isSymmetricHelper(p.left, q.right) && isSymmetricHelper(p.right, q.left);
    }

    //iterative way

    public boolean isSymmetricIterative(TreeNode root) {
        if (root == null || root.left == null && root.right == null)
            return true;
        Queue<TreeNode> q = new LinkedList<TreeNode>();
        //q.offer(root);
        if (root.left != null) q.offer(root.left);
        if (root.right != null) q.offer(root.right);
        while (!q.isEmpty()) {
            if (q.size() % 2 != 0)
                return false;
            TreeNode node1 = q.poll();
            TreeNode node2 = q.poll();
            if (node1.val != node2.val)
                return false;
            if (node1.left != null) {
                if (node2.right == null)
                    return false;
                q.offer(node1.left);
                q.offer(node2.right);
            } else if (node2.right != null) {
                return false;
            }
            if (node1.right != null) {
                if (node2.left == null)
                    return false;
                q.offer(node1.right);
                q.offer(node2.left);
            } else if (node2.left != null) {
                return false;
            }
        }
        return true;

    }

    public boolean isSymmetricIt(TreeNode root) {
        if (root == null || root.left == null && root.right == null)
            return true;
        Queue<TreeNode> q = new LinkedList<TreeNode>();
        //q.offer(root);
        if (root.left != null) q.offer(root.left);
        if (root.right != null) q.offer(root.right);
        while (!q.isEmpty()) {
            if (q.size() % 2 != 0)
                return false;
            TreeNode node1 = q.poll();
            TreeNode node2 = q.poll();
            if (node1.val != node2.val)
                return false;
            if (node1.left != null) {
                q.offer(node1.left);

            }
            if (node2.right != null) {
                q.offer(node2.right);
            }
            if (q.size() % 2 != 0)
                return false;
            if (node1.right != null) {
                q.offer(node1.right);
            }
            if (node2.left != null) {
                q.offer(node2.left);
            }
            if (q.size() % 2 != 0)
                return false;
        }
        return true;
    }


    //270 closest binary search tree value
    //recursive way
    //has duplicate
    /*
     int a = root.val;
    TreeNode kid = target < a ? root.left : root.right;
    if (kid == null) return a;
    int b = closestValue(kid, target);
    return Math.abs(a - target) < Math.abs(b - target) ? a : b;
     */
    public int closestValue(TreeNode root, double target) {
        if (root == null || root.left == null && root.right == null)
            return root == null ? Integer.MAX_VALUE : root.val;
        if (1.0 * root.val > target && root.left != null) {
            int left = closestValue(root.left, target);
            return Math.abs(root.val - target) > Math.abs(left - target) ? left : root.val;
        } else if (1.0 * root.val < target && root.right != null) {
            int right = closestValue(root.right, target);
            return Math.abs(root.val - target) > Math.abs(right - target) ? right : root.val;
        } else return root.val;
    }

    //closest binary search tree value iterative way
    //concise
    /*
     int closest = root->val;
    while (root) {
        if (abs(closest - target) >= abs(root->val - target))
            closest = root->val;
        root = target < root->val ? root->left : root->right;
    }
    return closest;
     */
    public int closestValueIterative(TreeNode root, double target) {
        //since there is always one solution
        int res = root.val;
        while (root != null) {
            if (Math.abs(res - target) > Math.abs(root.val - target))
                res = root.val;
            if (1.0 * root.val > target && root.left != null)
                root = root.left;
            else if (1.0 * root.val < target && root.right != null)
                root = root.right;
            else break;
        }
        return res;
    }

    //272 cloestBinarySearchValue II

    public TreeNode getPredecessor(TreeNode root, TreeNode node) {
        if (root == null)
            return null;
        if (root.val >= node.val)
            return getPredecessor(root.left, node);
        else {
            TreeNode right = getPredecessor(root.right, node);
            return right != null ? right : root;
        }
    }

    public TreeNode getSuccessor(TreeNode root, TreeNode node) {
        if (root == null)
            return null;
        if (root.val <= node.val)
            return getSuccessor(root.right, node);
        else {
            TreeNode left = getSuccessor(root.left, node);
            return left != null ? left : root;
        }
    }

    public List<Integer> closestKValues(TreeNode root, double target, int k) {
        TreeNode closest = root;
        TreeNode saveRoot = root;
        while (saveRoot != null) {
            if (Math.abs(closest.val - target) >= Math.abs(saveRoot.val - target))
                closest = saveRoot;
            saveRoot = target < saveRoot.val ? saveRoot.left : saveRoot.right;
        }
        List<Integer> res = new ArrayList<Integer>();
        res.add(closest.val);
        TreeNode next = getSuccessor(root, closest);
        TreeNode preNode = getPredecessor(root, closest);
        while (res.size() != k) {
            if (next == null || preNode == null) {
                res.add(next != null ? next.val : preNode.val);
                next = next != null ? getSuccessor(root, next) : null;
                preNode = preNode != null ? getPredecessor(root, preNode) : null;
                if (res.size() == k)
                    break;
            } else {
                if (Math.abs(next.val - target) > Math.abs(preNode.val - target)) {
                    res.add(preNode.val);
                    preNode = getPredecessor(root, preNode);
                } else {
                    res.add(next.val);
                    next = getSuccessor(root, next);
                }
                if (res.size() == k)
                    break;
            }
        }
        return res;
    }


    //inorder traversal to split the inorder array into two parts

    public void inorder(TreeNode root, double target, boolean reverse, Stack<Integer> stk) {
        if (root == null)
            return;
        inorder(reverse ? root.right : root.left, target, reverse, stk);
        if ((reverse && root.val <= target) || (!reverse && root.val > target))
            return;
        stk.push(root.val);
        inorder(reverse ? root.left : root.right, target, reverse, stk);
    }

    public List<Integer> closestKValuesSplitTwoParts(TreeNode root, double target, int k) {
        List<Integer> res = new ArrayList<Integer>();
        Stack<Integer> s1 = new Stack<Integer>();
        Stack<Integer> s2 = new Stack<Integer>();
        inorder(root, target, false, s1);
        inorder(root, target, true, s2);
        while (k-- > 0) {
            if (s1.empty()) {
                res.add(s2.pop());
            } else if (s2.empty()) {
                res.add(s1.pop());
            } else if (Math.abs(s1.peek() - target) < Math.abs(s2.peek() - target)) {
                res.add(s1.pop());
            } else
                res.add(s2.pop());
        }
        return res;
    }





    //112 path sum
    public boolean hasPathSum(TreeNode root, int sum) {
        if (root == null)
            return false;
        if (root.left == null && root.right == null && sum == root.val)
            return true;
        return root.left != null && hasPathSum(root.left, sum - root.val) || root.right != null && hasPathSum(root.right, sum - root.val);
    }

    //iterative way
    //in case of root and node
    public boolean hasPathSumIterative(TreeNode root, int sum) {
        if (root == null)
            return false;
        Queue<Tuple<TreeNode, Integer>> q = new LinkedList<Tuple<TreeNode, Integer>>();
        q.offer(new Tuple<TreeNode, Integer>(root, new Integer(root.val)));
        while (!q.isEmpty()) {
            Tuple t = q.poll();
            TreeNode node = (TreeNode) t.x;
            int val = ((Integer) t.y).intValue();
            if (node.left == null && node.right == null && val == sum)
                return true;
            if (node.left != null) {
                q.offer(new Tuple(node.left, new Integer(val + node.left.val)));
            }
            if (node.right != null) {
                q.offer(new Tuple(node.right, new Integer(val + node.right.val)));
            }
        }
        return false;
    }

    //you can use the postorder
    public boolean hasPathSumIterativePostorder(TreeNode root, int sum) {
        Stack<TreeNode> stk = new Stack<TreeNode>();
        TreeNode pre = null;
        TreeNode cur = root;
        int SUM = 0;
        while (cur != null || !stk.isEmpty()) {
            while (cur != null) {
                stk.push(cur);
                SUM += cur.val;
                cur = cur.left;
            }
            cur = stk.peek();
            if (cur.left == null && cur.right == null && SUM == sum) {
                return true;
            }
            if (cur.right != null && pre != cur.right) {
                cur = cur.right;
            } else {//if there is no right child or right child has been visited, then you can start to visit the part
                //System.out.println(cur.val);
                pre = cur;
                stk.pop();
                SUM -= cur.val;
                cur = null;
            }
        }
        return false;
    }

    //113 path sum II

    public void pathSumDfs(TreeNode root, List<List<Integer>> res, List<Integer> path, int sum) {
        if (root == null)
            return;
        path.add(root.val);
        if (root.left == null && root.right == null && sum == root.val) {


            List<Integer> newPath = new ArrayList<Integer>(path);
            res.add(newPath);
            path.remove(path.size() - 1);//do not forget this sentence, otherwise,you can delete return
            return;
        }
        if (root.left != null) {
            pathSumDfs(root.left, res, path, sum - root.val);
        }
        if (root.right != null) {
            pathSumDfs(root.right, res, path, sum - root.val);
        }
        path.remove(path.size() - 1);
    }

    public List<List<Integer>> pathSum(TreeNode root, int sum) {
        List<List<Integer>> res = new ArrayList<List<Integer>>();
        List<Integer> path = new ArrayList<Integer>();
        pathSumDfs(root, res, path, sum);
        return res;
    }


    //iterative way
    //level order
    public List<List<Integer>> PathSumIIterative(TreeNode root, int sum) {
        Map<TreeNode, TreeNode> parent = new HashMap<TreeNode, TreeNode>();
        List<List<Integer>> res = new ArrayList<List<Integer>>();
        if (root == null)
            return res;
        Queue<TreeNode> q = new LinkedList<TreeNode>();
        q.offer(root);
        while (!q.isEmpty()) {
            TreeNode node = q.poll();
            if (node.left == null && node.right == null) {
                //this is a leaf node
                //find its parent one by one
                TreeNode tmp = node;
                int SUM = node.val;
                List<Integer> path = new ArrayList<Integer>();
                path.add(node.val);
                while (parent.containsKey(tmp)) {
                    SUM += parent.get(tmp).val;
                    path.add(0, parent.get(tmp).val);
                    tmp = parent.get(tmp);
                }
                if (SUM == sum)
                    res.add(path);
            }
            if (node.left != null) {
                parent.put(node.left, node);
                q.offer(node.left);
            }
            if (node.right != null) {
                parent.put(node.right, node);
                q.offer(node.right);
            }
        }
        return res;
    }

    //path sumII postorder iterative way
    public List<List<Integer>> PathSumIIIterativePostorder(TreeNode root, int sum) {
        Stack<TreeNode> stk = new Stack<TreeNode>();
        TreeNode pre = null;
        TreeNode cur = root;
        List<List<Integer>> res = new ArrayList<List<Integer>>();
        int SUM = 0;
        while (cur != null || !stk.isEmpty()) {
            while (cur != null) {
                stk.push(cur);
                cur = cur.left;
                SUM += cur.val;
            }
            cur = stk.peek();
            if (cur.left == null && cur.right == null && SUM == sum) {
                List<Integer> path = new ArrayList<Integer>();
                for (TreeNode node : stk) {
                    path.add(node.val);
                }
                res.add(path);
            }
            if (cur.right != null && pre != cur.right) {
                cur = cur.right;
            } else {//if there is no right child or right child has been visited, then you can start to visit the part
                //System.out.println(cur.val);
                pre = cur;
                stk.pop();
                SUM -= cur.val;
                cur = null;
            }
        }
        return res;
    }


    //437 path sum III
    //dfs enumerate all possible start points
    //then dfs from the start point
    //interesting solution
    int number = 0;

    public void pathSumIIIDfs1(TreeNode root, int sum) {
        if (root == null)
            return;
        if (root.val == sum) {
            number++;
        }
        pathSumIIIDfs1(root.left, sum - root.val);
        pathSumIIIDfs1(root.right, sum - root.val);
    }

    public void pathSumIIIDfs(TreeNode root, int sum) {
        if (root == null)
            return;
        pathSumIIIDfs1(root, sum);
        pathSumIIIDfs(root.left, sum);
        pathSumIIIDfs(root.right, sum);
    }

    public int pathSumIII(TreeNode root, int sum) {
        if (root == null)
            return 0;
        pathSumIIIDfs(root, sum);
        return number;
    }




    //404 sum of left leaves
    int leafsum = 0;

    public void sumOfLeftLeavesDfs(TreeNode root) {
        if (root == null)
            return;
        if (root.left != null && root.left.left == null && root.left.right == null)
            leafsum += root.left.val;
        sumOfLeftLeavesDfs(root.left);
        sumOfLeftLeavesDfs(root.right);
    }

    public int sumOfLeftLeaves(TreeNode root) {
        if (root == null)
            return 0;
        sumOfLeftLeavesDfs(root);
        return leafsum;
    }

    //recursive way
    //interesting

    public int sumOfLeftLeavesRecursive(TreeNode root) {
        if (root == null)
            return 0;
        int ans = 0;
        if (root.left != null) {
            if (root.left.left == null && root.left.right == null)
                ans += root.left.val;
            else
                ans += sumOfLeftLeavesRecursive(root.left);
        }
        ans += sumOfLeftLeavesRecursive(root.right);
        return ans;
    }

    //iterative way
    public int sumOfLeftLeavesIterative(TreeNode root) {
        if (root == null)
            return 0;
        Stack<TreeNode> stk = new Stack<TreeNode>();
        stk.push(root);
        int ans = 0;
        while (!stk.isEmpty()) {
            TreeNode node = stk.pop();
            if (node.left != null) {
                if (node.left.left == null && node.left.right == null)
                    ans += node.left.val;
                else
                    stk.push(node.left);
            }
            if (node.right != null) {
                stk.push(node.right);
            }
        }
        return ans;
    }

    //leetcode 222 count complete Tree nodes
    public int getLeft(TreeNode node) {
        int count = 0;
        while (node != null) {
            count++;
            node = node.left;
        }
        return count;
    }

    public int getRight(TreeNode node) {
        int count = 0;
        while (node != null) {
            count++;
            node = node.right;
        }
        return count;
    }

    //So overall O(log(n)^2).
    //I have O(log(n)) steps. Finding a height costs O(log(n))
    public int countNodes(TreeNode root) {
        if (root == null)
            return 0;
        int ldepth = getLeft(root.left);
        int rdepth = getRight(root.right);
        return ldepth == rdepth ? (1 << ldepth) - 1 : countNodes(root.left) + countNodes(root.right);
    }

    //you can also use isbalanced to solve
    public int BalancedHeight(TreeNode root) {
        if (root == null)
            return 0;
        int left = BalancedHeight(root.left);
        int right = BalancedHeight(root.right);
        if (left < 0 || right < 0 || left != right)
            return -1;
        else return left + 1;
    }

    public int countNodesAnother(TreeNode root) {
        if (root == null)
            return 0;
        int num = BalancedHeight(root);
        return num == -1 ? 1 + countNodesAnother(root.left) + countNodesAnother(root.right) : (1 << num) - 1;//(1<<num)-1 can't write the 1<<num-1;
    }




    //count the last level of complement tree

    public int countNodesLastLevel(TreeNode root) {
        if (root == null)
            return 0;
        if (root.left == null)
            return 1;
        int height = 0;
        int nodeSum = 0;
        TreeNode curr = root;
        while (curr.left != null) {
            nodeSum += (1 << height);
            height++;
            curr = curr.left;
        }
        return nodeSum + countLastLevel(root, height);
    }

    public int countLastLevel(TreeNode root, int height) {
        if (height == 1) {
            if (root.right != null)
                return 2;
            else if (root.left != null)
                return 1;
            else
                return 0;
        }
        TreeNode midNode = root.left;
        int currHeight = 1;
        while (currHeight < height) {
            currHeight++;
            midNode = midNode.right;
        }
        if (midNode == null)
            return countLastLevel(root.left, height - 1);
        else return (1 << (height - 1)) + countLastLevel(root.right, height - 1);
    }

    public List<List<Integer>> zigzagLevelOrder(TreeNode root) {
        List<List<Integer>> res = new LinkedList<List<Integer>>();
        if (root == null)
            return res;
        Queue<TreeNode> q = new LinkedList<TreeNode>();
        q.offer(root);
        while (!q.isEmpty()) {
            int size = q.size();
            List<Integer> inner = new ArrayList<Integer>();
            boolean isReverse = res.size() % 2 == 1;
            while (size-- > 0) {
                TreeNode node = q.poll();
                if (isReverse)
                    inner.add(0, node.val);
                else
                    inner.add(node.val);
                if (node.left != null)
                    q.offer(node.left);
                if (node.right != null)
                    q.offer(node.right);
            }
            res.add(inner);
        }
        return res;
    }

    //dfs version
    public void dfs(TreeNode root, List<List<Integer>> res, int height) {
        if (root == null) return;
        if (res.size() == height) {
            List<Integer> inner = new LinkedList<Integer>();
            res.add(inner);
        }
        if (height % 2 == 1) res.get(height).add(0, root.val);
        else res.get(height).add(root.val);

        dfs(root.left, res, height + 1);
        dfs(root.right, res, height + 1);
    }

    public List<List<Integer>> zigzagLevelOrderDfsVersion(TreeNode root) {
        List<List<Integer>> res = new LinkedList<List<Integer>>();
        dfs(root, res, 0);
        return res;
    }


    //129. Sum Root to Leaf Numbers
    int sumRes = 0;

    public void dfs(TreeNode root, int sum1) {
        if (root == null) return;
        if (root.left == null && root.right == null) {
            sumRes += sum1;
        }
        if (root.left != null) dfs(root.left, 10 * sum1 + root.left.val);
        if (root.right != null) dfs(root.right, 10 * sum1 + root.right.val);
    }

    public int sumNumbers(TreeNode root) {
        if (root == null) return 0;
        dfs(root, root.val);
        return sumRes;
    }

    //without instance varibale
    public int sumNumbersII(TreeNode root) {
        if (root == null)
            return 0;
        return sumR(root, 0);
    }

    public int sumR(TreeNode root, int x) {
        if (root.right == null && root.left == null)
            return 10 * x + root.val;
        int val = 0;
        if (root.left != null)
            val += sumR(root.left, 10 * x + root.val);
        if (root.right != null)
            val += sumR(root.right, 10 * x + root.val);
        return val;
    }

    //129 iterative way
    public int sumNumbersIterative(TreeNode root) {
        int res = 0;
        Queue<Tuple<TreeNode, Integer>> q = new LinkedList<Tuple<TreeNode, Integer>>();
        if (root == null)
            return res;
        q.offer(new Tuple<TreeNode, Integer>(root, new Integer(root.val)));
        while (!q.isEmpty()) {
            Tuple t = q.poll();
            TreeNode node = (TreeNode) t.x;
            int val = ((Integer) t.y).intValue();
            if (node.left == null && node.right == null)
                res += val;
            if (node.left != null) {
                q.offer(new Tuple(node.left, 10 * val + node.left.val));
            }
            if (node.right != null) {
                q.offer(new Tuple(node.right, 10 * val + node.right.val));
            }
        }
        return res;
    }

    //105, construct binary tree from preorder and inorder

    public TreeNode makeTree(int[] preorder, int begin1, int end1, int[] inorder, int begin2, int end2) {
        if (begin1 > end1)
            return null;
        if (begin1 == end1)
            return new TreeNode(preorder[begin1]);
        else {
            TreeNode root = new TreeNode(preorder[begin1]);
            int index = begin2;
            while (index < end2) {
                if (inorder[index] == preorder[begin1])
                    break;
                index++;
            }
            root.left = makeTree(preorder, begin1 + 1, index - begin2 + begin1, inorder, begin2, index - 1);
            root.right = makeTree(preorder, index - begin2 + begin1 + 1, end1, inorder, index + 1, end2);
            return root;
        }
    }

    public TreeNode buildTree(int[] preorder, int[] inorder) {
        int n = preorder.length;
        return makeTree(preorder, 0, n - 1, inorder, 0, n - 1);
    }

    //iterative way
    public TreeNode bulidTreeIterative(int[] preorder, int[] inorder) {
        if (preorder.length == 0)
            return null;
        Stack<TreeNode> stk = new Stack<TreeNode>();
        TreeNode root = new TreeNode(preorder[0]);
        TreeNode cur = root;
        for (int i = 1, j = 0; i < preorder.length; ++i) {
            if (cur.val != inorder[j]) {
                cur.left = new TreeNode(preorder[i]);
                stk.push(cur);
                cur = cur.left;
            } else {
                j++;
                while (!stk.isEmpty() && stk.peek().val == inorder[j]) {
                    cur = stk.pop();
                    j++;
                }
                cur = cur.right = new TreeNode(preorder[i]);
            }
        }
        return root;
    }


    //106 build tree from postorder and inorder
    //iterative way
    public TreeNode buildTreeIterativeForPostorder(int[] inorder, int[] postorder) {
        if (postorder.length == 0)
            return null;
        Stack<TreeNode> stk = new Stack<TreeNode>();
        int n = postorder.length;
        TreeNode root = new TreeNode(postorder[n - 1]);
        TreeNode cur = root;
        for (int i = n - 2, j = n - 1; i >= 0; --i) {
            if (cur.val != inorder[j]) {
                cur.right = new TreeNode(postorder[i]);
                stk.push(cur);
                cur = cur.right;
            } else {
                j--;
                while (!stk.isEmpty() && stk.peek().val == inorder[j]) {
                    cur = stk.pop();
                    j--;
                }
                cur = cur.left = new TreeNode(postorder[i]);
            }
        }
        return root;
    }

    //recursive way
    public TreeNode makeTreeRecursive(int[] inorder, int begin1, int end1, int[] postorder, int begin2, int end2) {
        if (begin2 > end2)
            return null;
        if (begin2 == end2)
            return new TreeNode(postorder[begin2]);
        else {
            TreeNode root = new TreeNode(postorder[end2]);
            int index = begin1;
            while (index < end1) {
                if (inorder[index] == postorder[end2])
                    break;
                index++;
            }
            root.left = makeTreeRecursive(inorder, begin1, index - 1, postorder, begin2, begin2 + index - 1 - begin1);
            root.right = makeTreeRecursive(inorder, index + 1, end1, postorder, index - begin1 + begin2, end2 - 1);
            return root;
        }

    }

    public TreeNode buildTreeRecursiveway(int[] inorder, int[] postorder) {
        int n = inorder.length;
        return makeTreeRecursive(inorder, 0, n - 1, postorder, 0, n - 1);
    }


    //199 binary tree right side view
    public List<Integer> rightSideView(TreeNode root) {
        List<Integer> res = new LinkedList<Integer>();
        Queue<TreeNode> q = new LinkedList<TreeNode>();
        if (root == null)
            return res;
        q.offer(root);
        while (!q.isEmpty()) {
            int size = q.size();
            while (size-- > 0) {
                TreeNode node = q.poll();
                if (size == 0)
                    res.add(node.val);
                if (node.left != null)
                    q.offer(node.left);
                if (node.right != null)
                    q.offer(node.right);
            }
        }
        return res;
    }

    //recursive way
    //the level order's change
    //interesting
    public void rightview(TreeNode node, List<Integer> res, int level) {
        if (node == null)
            return;
        if (level == res.size())
            res.add(node.val);
        rightview(node.right, res, level + 1);
        rightview(node.left, res, level + 1);
    }

    public List<Integer> rightSideViewRecursive(TreeNode node) {
        List<Integer> res = new ArrayList<Integer>();
        rightview(node, res, 0);
        return res;
    }
    //divide and conquer
    /*
    public List<Integer> rightSideView(TreeNode root) {
    if(root==null)
        return new ArrayList<Integer>();
    List<Integer> left = rightSideView(root.left);
    List<Integer> right = rightSideView(root.right);
    List<Integer> re = new ArrayList<Integer>();
    re.add(root.val);
    for(int i=0;i<Math.max(left.size(), right.size());i++){
        if(i>=right.size())
            re.add(left.get(i));
        else
            re.add(right.get(i));
    }
    return re;
}
     */

    //230 kth smallest element in a bst
    //bst
    //almost pass;//sigh,forget a lot ,you should change k too, otherwise,it will update res often;
    int res = 0;

    public int kthSmallestInorder(TreeNode root, int k) {
        if (root == null || k < 0)
            return k;
        k = kthSmallestInorder(root.left, k);
        k--;
        if (k == 0) {
            res = root.val;
            return k;
        }
        if (k > 0)
            k = kthSmallestInorder(root.right, k);
        return k;
    }

    public int kthSmallest(TreeNode root, int k) {
        kthSmallestInorder(root, k);
        return res;
    }

    //naive way was find the inorder array first and then return v[k-1] directly
    //iterative way
    //do you remember how to write the inorder traversal
    public int kthSmallestIterative(TreeNode root, int k) {
        TreeNode cur = root;
        TreeNode prev = null;
        while (cur != null) {
            if (cur.left == null) {
                k--;
                if (k == 0)
                    return cur.val;
                cur = cur.right;
            } else {
                //find predecessor
                prev = cur.left;
                while (prev.right != null && prev.right != cur) {
                    prev = prev.right;
                }
                if (prev.right == null) {
                    prev.right = cur;
                    cur = cur.left;
                } else {
                    prev.right = null;
                    k--;
                    if (k == 0)
                        return cur.val;
                    cur = cur.right;
                }
            }
        }
        return -1;
    }
    //binary search way
    /*
    public int kthSmallest(TreeNode root, int k) {
        int count = countNodes(root.left);
        if (k <= count) {
            return kthSmallest(root.left, k);
        } else if (k > count + 1) {
            return kthSmallest(root.right, k-1-count); // 1 is counted as current node
        }

        return root.val;
    }

    public int countNodes(TreeNode n) {
        if (n == null) return 0;
        return 1 + countNodes(n.left) + countNodes(n.right);
    }

     */

    //173 binary search tree iterator
    //interesting question
    //So this can satisfy O(h) memory, hasNext() in O(1) time,
    //But next() is O(h) time.
    // I can't find a solution that can satisfy both next() in O(1) time, space in O(h).
    //Each node will be visited at most twice-> average O(1) run time.
    //The average time complexity of next() function is O(1) indeed in your case.
    // As the next function can be called n times at most, and the number of right nodes in self.pushAll(tmpNode.right)
    // function is maximal n in a tree which has n nodes, so the amortized time complexity is O(1).
    public class BSTIterator {
        private Stack<TreeNode> stk;

        public BSTIterator(TreeNode root) {
            stk = new Stack<TreeNode>();
            findNext(root);

        }

        /**
         * @return whether we have a next smallest number
         */
        public boolean hasNext() {
            return stk.isEmpty();
        }

        /**
         * @return the next smallest number
         */
        public int next() {
            TreeNode cur = stk.pop();
            int val = cur.val;
            if (cur.right != null)
                findNext(cur.right);
            return val;
        }

        public void findNext(TreeNode node) {
            if (node == null)
                return;
            while (node != null) {
                stk.push(node);
                node = node.left;
            }
        }
    }

    //another version
    public class BSTIteratorAnother {
        Stack<TreeNode> stk = null;
        TreeNode current = null;

        public BSTIteratorAnother(TreeNode root) {
            current = root;
            stk = new Stack<TreeNode>();
        }

        public boolean hasNext() {
            return !stk.isEmpty() || current != null;
        }

        public int next() {
            while (current != null) {
                stk.push(current);
                current = current.left;
            }
            TreeNode t = stk.pop();
            current = t.right;
            return t.val;
        }
    }

    //morris traversal solution

    public class BSTIteratorMorris {
        private TreeNode cur = null;

        public BSTIteratorMorris(TreeNode node) {
            cur = node;
        }

        public boolean hasNext() {
            return cur != null;
        }

        public int next() {
            TreeNode prev = null;
            int res = 0;
            while (cur != null) {
                if (cur.left == null) {
                    res = cur.val;
                    cur = cur.right;
                    break;
                } else {
                    //find predecessor
                    prev = cur.left;
                    while (prev.right != null && prev.right != cur) {
                        prev = prev.right;
                    }
                    if (prev.right == null) {
                        prev.right = cur;
                        cur = cur.left;
                    } else {
                        prev.right = null;
                        res = cur.val;
                        cur = cur.right;
                        break;
                    }
                }
            }
            return res;
        }
    }

    //114 flatten binary tree to linked list
    //like preorder
    public void flatten(TreeNode root) {
        if (root == null)
            return;
        flatten(root.left);
        flatten(root.right);
        TreeNode save_right = root.right;
        TreeNode node = root.left;
        if (node == null)// or you can start from root, because root->leftpart->rightpart, the leftpart could be null, so start from root is right,you can also judge the left part, if the left part is null, you can return
            return;
        root.left = null;
        root.right = node;
        while (node.right != null)
            node = node.right;
        node.right = save_right;
    }

    //reverse preorder
    private TreeNode prevous = null;

    public void flattenReversePreorder(TreeNode root) {
        if (root == null)
            return;
        flattenReversePreorder(root.right);
        flattenReversePreorder(root.left);
        root.left = null;
        root.right = prevous;
        prevous = root;
    }

    public void flattenByLeft(TreeNode root) {
        if (root == null)
            return;
        flattenByLeft(root.left);
        flattenByLeft(root.right);
        root.left = prevous;
        root.right = null;
        prevous = root;
    }

    //But flatten() would be unreusable if prev is set to a field
    public void flattenWithoutInstanceVariable(TreeNode root) {
        flatten(root, null);
    }

    private TreeNode flatten(TreeNode root, TreeNode pre) {
        if (root == null) return pre;
        pre = flatten(root.right, pre);
        pre = flatten(root.left, pre);
        root.right = pre;
        root.left = null;
        pre = root;
        return pre;
    }

    //iterative way O(1) space
    //interesting
    void flattenIteative(TreeNode root) {
        TreeNode curr = root;
        while (curr != null) {
            if (curr.left != null) {
                //find current node's prenode that links to current nodes' right subtree
                TreeNode pre = curr.left;
                while (pre.right != null) {
                    pre = pre.right;
                }
                pre.right = curr.right;
                //use current node's left subtree to replace its right subtree
                //subtree is already linked by current node prenode;
                curr.right = curr.left;
                curr.left = null;
            }
            curr = curr.right;
        }
    }

    //285 inorder successor in BST
    public TreeNode inorderSuccessor(TreeNode root, TreeNode p) {
        if (root == null || p == null)
            return null;
        if (root.val <= p.val)
            return inorderSuccessor(root.right, p);
        else {
            TreeNode left = inorderSuccessor(root.left, p);
            return left != null ? left : root;
        }
    }

    public TreeNode inorderSuccessorIterative(TreeNode root, TreeNode p) {
        if (root == null || p == null)
            return null;
        TreeNode successor = null;
        if (p.right != null) {
            successor = p.right;
            while (successor.left != null) {
                successor = successor.left;
            }
            return successor;
        }
        while (root != null) {
            if (root.val > p.val) {
                successor = root;
                root = root.left;
            } else if (root.val < p.val) {
                root = root.right;
            } else break;
        }
        return successor;
    }

    //The time complexity should be O(h)
    public TreeNode inorderSuccessorIterative2(TreeNode root, TreeNode p) {
        if (root == null || p == null)
            return null;
        TreeNode successor = null;
        while (root != null) {
            if (root.val <= p.val) {
                root = root.right;
            } else {
                successor = root;
                root = root.left;
            }
        }
        return successor;
    }

    //predecessor
    public TreeNode predecessor(TreeNode root, TreeNode p) {
        if (root == null || p == null)
            return null;
        if (root.val >= p.val)
            return predecessor(root.left, p);
        else {
            TreeNode right = predecessor(root.right, p);
            return right != null ? right : root;
        }
    }

    //iterative way
    public TreeNode predecessorIterative(TreeNode root, TreeNode p) {
        if (root == null || p == null)
            return null;
        TreeNode predecessor = null;
        while (root != null) {
            if (root.val >= p.val) {
                root = root.left;
            } else {
                predecessor = root;
                root = root.right;
            }
        }
        return predecessor;
    }


    //98 validate binary search tree
    //you can use inorder traversal to get the array and judge the array
    //recursive way
    public boolean isValidBST(TreeNode root) {
        if (root == null)
            return true;
        if (!isValidBST(root.left))
            return false;
        if (prevous != null && prevous.val >= root.val)
            return false;
        prevous = root;
        return isValidBST(root.right);
    }

    //another recursive way

    public boolean valid(TreeNode root, Integer min, Integer max) {
        if (root == null)
            return true;
        if ((min != null && min >= root.val) || (max != null && max <= root.val))
            return false;
        return valid(root.left, min, root.val) && valid(root.right, root.val, max);
    }

    public boolean isValidBSTRecursive(TreeNode root) {
        return valid(root, null, null);
    }

    //iterative way

    public boolean isValidBSTIterative(TreeNode root) {
        TreeNode cur = root;
        TreeNode prev = null;
        TreeNode pNode = null;//the previous node of inorder traversal
        while (cur != null) {
            if (cur.left == null) {
                if (pNode != null && pNode.val >= cur.val)
                    return false;
                pNode = cur;
                cur = cur.right;
            } else {
                //find predecessor
                prev = cur.left;
                while (prev.right != null && prev.right != cur) {
                    prev = prev.right;
                }
                if (prev.right == null) {
                    prev.right = cur;
                    cur = cur.left;
                } else {
                    prev.right = null;
                    if (pNode != null && pNode.val >= cur.val)
                        return false;
                    pNode = cur;
                    cur = cur.right;
                }
            }
        }
        return true;
    }


    //95 unqiue binary search TREE II


    //interesting
    public List<TreeNode> makeUniqueTrees(int begin, int end) {
        List<TreeNode> res = new ArrayList<TreeNode>();
        if (begin > end)
            res.add(null);
        else if (begin == end) {
            res.add(new TreeNode(begin));
        } else {
            for (int i = begin; i <= end; ++i) {
                List<TreeNode> left = makeUniqueTrees(begin, i - 1);
                List<TreeNode> right = makeUniqueTrees(i + 1, end);
                for (int k = 0; k < left.size(); ++k) {
                    for (int j = 0; j < right.size(); ++j) {
                        TreeNode root = new TreeNode(i);
                        root.left = left.get(k);
                        root.right = right.get(j);
                        res.add(root);
                    }
                }
            }
        }
        return res;
    }

    public List<TreeNode> generateTrees(int n) {
        if (n < 1)
            return new ArrayList<TreeNode>();
        return makeUniqueTrees(1, n);
    }

    //96 unique binary search tree
    public int numTrees(int n) {
        //cataland number
        int[] dp = new int[n + 1];
        dp[0] = dp[1] = 1;
        for (int i = 2; i <= n; ++i) {
            for (int j = 0; j <= i - 1; ++j) {
                dp[i] += dp[j] * dp[i - 1 - j];
            }
        }
        return dp[n];
    }


    //337 house robber III
    //naive
    public int robIIINaive(TreeNode root) {
        if (root == null)
            return 0;
        int val = 0;
        if (root.left != null) {
            val += robIIINaive(root.left.left) + robIIINaive(root.left.right);
        }
        if (root.right != null) {
            val += robIIINaive(root.right.left) + robIIINaive(root.right.right);
        }
        return Math.max(root.val + val, robIIINaive(root.left) + robIIINaive(root.right));
    }

    //step II--memory based

    public int robIIIStep(TreeNode root) {
        return robsub(root, new HashMap<TreeNode, Integer>());
    }

    public int robsub(TreeNode root, Map<TreeNode, Integer> map) {
        if (root == null)
            return 0;
        if (map.containsKey(root))
            return map.get(root);
        int val = 0;
        if (root.left != null) {
            val += robsub(root.left.left, map) + robsub(root.left.right, map);
        }
        if (root.right != null) {
            val += robsub(root.right.right, map) + robsub(root.right.left, map);
        }
        val = Math.max(val + root.val, robsub(root.left, map) + robsub(root.right, map));
        map.put(root, val);
        return val;
    }


    //optimal
    public int[] robIIIDfs(TreeNode root) {
        int[] res = {0, 0};
        if (root == null)
            return res;
        res[0] = res[1] = Integer.MIN_VALUE;
        int[] l = robIIIDfs(root.left);//res[0] contains  root, res[1] does not contain root
        int[] r = robIIIDfs(root.right);
        res[0] = l[1] + r[1] + root.val;
        res[1] = Math.max(l[0], l[1]) + Math.max(r[0], r[1]);//very interesting, can start from 3 level
        return res;
    }

    public int robIII(TreeNode root) {
        if (root == null)
            return 0;
        int[] res = robIIIDfs(root);
        return Math.max(res[0], res[1]);
    }


    //convert sorted array to binary search tree

    public TreeNode makeTreeFromArray(int[] nums, int begin, int end) {
        if (begin > end)
            return null;
        if (begin == end)
            return new TreeNode(nums[begin]);
        int mid = begin + (end - begin) / 2;
        TreeNode root = new TreeNode(nums[mid]);
        root.left = makeTreeFromArray(nums, begin, mid - 1);
        root.right = makeTreeFromArray(nums, mid + 1, end);
        return root;
    }

    public TreeNode sortedArrayToBST(int[] nums) {
        int n = nums.length;
        return makeTreeFromArray(nums, 0, n - 1);
    }

    public TreeNode sortedArrayToBSTIterative(int[] nums) {
        if (nums == null || nums.length == 0)
            return null;
        Queue<MyNode> q = new LinkedList<MyNode>();
        int left = 0;
        int right = nums.length - 1;
        int val = nums[left + (right - left) / 2];
        TreeNode root = new TreeNode(val);
        q.offer(new MyNode(root, left, right));
        while (!q.isEmpty()) {
            int size = q.size();
            while (size-- > 0) {
                MyNode cur = q.poll();
                int mid = cur.lb + (cur.rb - cur.lb) / 2;
                if (mid != cur.lb) {
                    TreeNode leftChild = new TreeNode(nums[cur.lb + (mid - 1 - cur.lb) / 2]);
                    cur.node.left = leftChild;
                    q.offer(new MyNode(leftChild, cur.lb, mid - 1));
                }
                if (mid != cur.rb) {
                    TreeNode rightChild = new TreeNode(nums[mid + 1 + (cur.rb - mid - 1) / 2]);
                    cur.node.right = rightChild;
                    q.offer(new MyNode(rightChild, mid + 1, cur.rb));
                }
            }
        }
        return root;
    }

    //156 binary tree upside down
    //Java recursive (O(logn) space) and iterative solutions (O(1) space)
    public TreeNode upsideDownBinaryTree(TreeNode root) {
        if (root == null || root.left == null && root.right == null)
            return root;
        TreeNode left = upsideDownBinaryTree(root.left);
        root.left.left = root.right;
        root.left.right = root;
        root.left = null;
        root.right = null;
        return left;
    }

    //iterative way

    public TreeNode upsideDownBinaryTreeIterative(TreeNode root) {
        TreeNode curr = root;
        TreeNode next = null;
        TreeNode prev = null;
        TreeNode tmp = null;//need temp to keep the previous right child;
        while (curr != null) {
            next = curr.left;
            curr.left = tmp;
            tmp = curr.right;
            curr.right = prev;
            prev = curr;
            curr = next;
        }
        return prev;
    }


    //366 find leaves of binary tree

    //recursive way
    //change tree's structure
    //worst case when the tree is a line
    public TreeNode dfsFindLeaves(TreeNode root, List<Integer> path) {
        if (root == null)
            return null;
        if (root.left == null && root.right == null) {
            path.add(root.val);
            return null;
        }
        root.left = dfsFindLeaves(root.left, path);
        root.right = dfsFindLeaves(root.right, path);
        return root;
    }

    public List<List<Integer>> findLeaves(TreeNode root) {
        List<List<Integer>> res = new ArrayList<List<Integer>>();
        List<Integer> path = new ArrayList<Integer>();
        while (root != null) {
            root = dfsFindLeaves(root, path);
            res.add(new ArrayList(path));
            path.clear();
        }
        return res;
    }

    //another recursive way
    public List<List<Integer>> findLeavesRecursive(TreeNode root) {
        List<List<Integer>> res = new ArrayList<List<Integer>>();
        height(root, res);
        return res;
    }

    int height(TreeNode node, List<List<Integer>> res) {
        if (node == null)
            return -1;
        int level = 1 + Math.max(height(node.left, res), height(node.right, res));
        if (res.size() == level)
            res.add(new ArrayList<Integer>());
        res.get(level).add(node.val);//one line node.left = node.right = null; to remove visited nodes
        return level;
    }
    //333 largest bst substree

    public info BSTdfs(TreeNode root) {
        if (root == null)
            return new info();
        info left = BSTdfs(root.left);
        info right = BSTdfs(root.right);
        info m = new info();
        if (!left.isBST || !right.isBST || root.val >= right.minValue || root.val <= left.maxValue) {
            m.size = Math.max(left.size, right.size);
            m.isBST = false;
        } else {
            m.size = left.size + right.size + 1;
            m.isBST = true;
            m.minValue = root.left != null ? left.minValue : root.val;
            m.maxValue = root.right != null ? right.maxValue : root.val;
        }
        return m;

    }

    public int largestBSTSubtree(TreeNode root) {
        return BSTdfs(root).size;
    }

    //255 verify preorder sequence in binary search tree
    public boolean verifyPreorder(int[] preorder) {
        if (preorder == null || preorder.length == 0)
            return true;
        int minValue = Integer.MIN_VALUE;
        int index = -1;
        for (int i = 0; i < preorder.length; ++i) {
            if (preorder[i] < minValue)
                return false;
            while (index >= 0 && preorder[i] >= preorder[index]) {
                minValue = preorder[index--];
            }
            preorder[++index] = preorder[i];
        }
        return true;
    }

    //by stack
    public boolean verifyPreorderByStack(int[] preorder) {
        int n = preorder.length;
        Stack<Integer> stk = new Stack<>();
        int minValue = Integer.MIN_VALUE;
        for (int i = 0; i < n; ++i) {
            if (preorder[i] < minValue)
                return false;
            while (!stk.isEmpty() && preorder[i] > stk.peek()) {
                minValue = stk.pop();
            }
            stk.push(preorder[i]);
        }
        return true;
    }


    //verify postorder sequence in bst
    public boolean verifyPostorder(int[] postorder) {
        if (postorder == null || postorder.length == 0)
            return true;
        int index = postorder.length;
        int maxValue = Integer.MAX_VALUE;
        for (int i = postorder.length - 1; i >= 0; --i) {
            if (postorder[i] > maxValue)
                return false;
            while (index < postorder.length && postorder[i] <= postorder[index]) {
                maxValue = postorder[index++];
            }
            postorder[--index] = postorder[i];
        }
        return true;
    }


    //450 delete node in bst
    public TreeNode deleteNode(TreeNode root, int key) {
        if (root == null)
            return null;
        if (root.val > key)
            root.left = deleteNode(root.left, key);
        else if (root.val < key)
            root.right = deleteNode(root.right, key);
        else {
            if (root.left == null && root.right == null)
                return null;
            else if (root.left == null) {
                root = root.right;
                return root;
            } else if (root.right == null) {
                root = root.left;
                return root;
            } else {
                TreeNode node = root.right;
                while (node.left != null) {
                    node = node.left;
                }
                root.val = node.val;
                root.right = deleteNode(root.right, root.val);
            }
        }
        return root;
    }

    private TreeNode deleteRootNode(TreeNode root) {
        if (root == null)
            return null;
        if (root.left == null)
            return root.right;
        if (root.right == null)
            return root.left;
        TreeNode next = root.right;
        TreeNode pre = null;
        while (next.left != null) {
            pre = next;
            next = next.left;
        }
        next.left = root.left;
        if (root.right != next) {//very interesting
            pre.left = next.right;
            next.right = root.right;
        }
        return next;

    }

    public TreeNode deleteNodeIterative(TreeNode root, int key) {
        TreeNode cur = root;
        TreeNode pre = null;
        while (cur != null && cur.val != key) {
            pre = cur;
            if (key < cur.val)
                cur = cur.left;
            else if (key > cur.val)
                cur = cur.right;
        }
        if (pre == null) {
            return deleteRootNode(cur);
        }
        if (pre.left == cur) {
            pre.left = deleteRootNode(cur);
        } else {
            pre.right = deleteRootNode(cur);
        }
        return root;
    }


    //298 longest consecutive sequence
    public void longestConsecutiveDfs(TreeNode root, int count, int val) {
        if (root == null)
            return;
        if (root.val == val + 1)
            count++;
        else
            count = 1;
        sumRes = Math.max(sumRes, count);
        longestConsecutiveDfs(root.left, count, root.val);
        longestConsecutiveDfs(root.right, count, root.val);
    }

    public int longestConsecutive(TreeNode root) {
        if (root == null)
            return 0;
        longestConsecutiveDfs(root, 0, root.val);
        return sumRes;
    }

    //without using instance variable
    //maybe that means if you want to examine another tree root, you should initialize a new object of class Solution. Because a global var max would influence this.
    public int longestConsecutiveRecursive(TreeNode root) {
        if (root == null) {
            return 0;
        }
        return DFS(root, root.val + 1, 1, 1);
    }

    private int DFS(TreeNode node, int target, int curr, int max) {
        if (node == null) {
            return max;
        }
        if (node.val == target) {
            curr++;
            max = Math.max(max, curr);
        } else {
            curr = 1;
        }
        return Math.max(DFS(node.left, node.val + 1, curr, max), DFS(node.right, node.val + 1, curr, max));
    }

    // iterative way
    //wrong solution
    //1,2,2 can not pass the test case
    //count would change, you should let count into the q
    public int longestConsecutiveIterative1(TreeNode root) {
        if (root == null) {
            return 0;
        }
        int res = 1;
        int count = 0;
        Queue<Tuple<TreeNode, Integer>> q = new LinkedList<Tuple<TreeNode, Integer>>();
        q.offer(new Tuple(root, root.val));
        while (!q.isEmpty()) {
            Tuple t = q.poll();
            TreeNode node = (TreeNode) t.x;
            int val = ((Integer) t.y).intValue();
            if (val == node.val) {
                count++;
                res = Math.max(count, res);
            } else
                count = 1;
            if (node.left != null) {
                q.offer(new Tuple(node.left, node.val + 1));
            }
            if (node.right != null) {
                q.offer(new Tuple(node.right, node.val + 1));
            }
        }
        return res;
    }


    //this is the right version
    public int longestConsecutiveIterative(TreeNode root) {
        if (root == null) {
            return 0;
        }
        int res = 1;
        Queue<Tuple<TreeNode, Integer>> q = new LinkedList<Tuple<TreeNode, Integer>>();
        q.offer(new Tuple(root, 1));
        while (!q.isEmpty()) {
            Tuple t = q.poll();
            TreeNode node = (TreeNode) t.x;
            int val = ((Integer) t.y).intValue();
            res = Math.max(val, res);
            if (node.left != null) {
                int count = node.left.val == node.val + 1 ? val + 1 : 1;
                q.offer(new Tuple(node.left, count));
            }
            if (node.right != null) {
                int count = node.right.val == node.val + 1 ? val + 1 : 1;
                q.offer(new Tuple(node.right, count));
            }
        }
        return res;
    }


    //116 Populating Next Right Pointers in Each Node
    //first O(N) space, using queue,level order
    public void connectI(TreeLinkNode root) {
        if (root == null)
            return;
        Queue<TreeLinkNode> q = new LinkedList<TreeLinkNode>();
        q.offer(root);
        while (!q.isEmpty()) {
            int size = q.size();
            while (size-- > 0) {
                TreeLinkNode node = q.poll();
                if (size != 0)
                    node.next = q.peek();
                if (node.left != null) {
                    q.offer(node.left);
                }
                if (node.right != null) {
                    q.offer(node.right);
                }
            }
        }
    }

    //O(1) space,
    public void connectO1Space(TreeLinkNode root) {
        if (root == null)
            return;
        TreeLinkNode cur = root;
        TreeLinkNode nextLinkNode = null;
        while (cur.left != null) {
            nextLinkNode = cur.left;
            while (cur != null) {
                cur.left.next = cur.right;
                if (cur.next != null)
                    cur.right.next = cur.next.left;
                cur = cur.next;
            }
            cur = nextLinkNode;
        }
        return;
    }

    //117 populating next right pointers in each node II
    //first O(N) space, using queue,level order
    public void connectO1SpaceII(TreeLinkNode root) {
        TreeLinkNode nextHead = new TreeLinkNode(0);
        nextHead.next = root;
        while (nextHead.next != null) {
            TreeLinkNode tail = nextHead;
            TreeLinkNode n = nextHead.next;
            nextHead.next = null;//go to next level
            for (; n != null; n = n.next) {
                if (n.left != null) {
                    tail.next = n.left;
                    tail = tail.next;
                }
                if (n.right != null) {
                    tail.next = n.right;
                    tail = tail.next;
                }
            }
        }
    }


    //124 Binary Tree Maximum Path Sum


    //return contains root's path, straight
    //computes the maximum path sum with highest node is the input node, update maximum if necessary
    public int dfsMaxPathSum(TreeNode root) {
        if (root == null)
            return 0;
        int l = dfsMaxPathSum(root.left);
        int r = dfsMaxPathSum(root.right);
        int res = Math.max(l + root.val, Math.max(r + root.val, root.val));
        sumRes = Math.max(res, Math.max(sumRes, root.val + l + r));
        return res;
    }

    public int maxPathSum(TreeNode root) {
        sumRes = Integer.MIN_VALUE;
        dfsMaxPathSum(root);
        return sumRes;
    }

    //250 count univalue subtrees

    public uniInfo findUniq(TreeNode root) {
        uniInfo res = new uniInfo();
        if (root == null)
            return res;
        if (root.left == null && root.right == null) {
            res.size = 1;
            res.isUni = true;
            res.value = root.val;
            return res;
        }
        uniInfo left = findUniq(root.left);
        uniInfo right = findUniq(root.right);
        if (left.size == 0 || right.size == 0) {
            if (left.size != 0)
                right.value = left.value;
            else
                left.value = right.value;
        }
        if (!left.isUni || !right.isUni || !(root.val == left.value && root.val == right.value)) {
            res.isUni = false;
            res.value = root.val;
            res.size = left.size + right.size;
        } else {
            res.value = root.val;
            res.isUni = true;
            res.size = left.size + right.size + 1;
        }
        return res;
    }

    public int countUnivalSubtrees(TreeNode root) {
        if (root == null)
            return 0;
        return findUniq(root).size;
    }

    //concise
    public int countUnivalSubtreesConcise(TreeNode root) {
        int[] count = new int[1];
        helper(root, count);
        return count[0];
    }

    private boolean helper(TreeNode node, int[] count) {
        if (node == null) {
            return true;
        }
        boolean left = helper(node.left, count);
        boolean right = helper(node.right, count);
        if (left && right) {
            if (node.left != null && node.val != node.left.val) {
                return false;
            }
            if (node.right != null && node.val != node.right.val) {
                return false;
            }
            count[0]++;
            return true;
        }
        return false;
    }

    //508 most frequence subtree sum
    public int dfsSum(TreeNode root, List<Integer> res) {
        if (root.left == null && root.right == null) {
            res.add(root.val);
            return root.val;
        }
        int val = root.val;
        if (root.left != null) {
            int leftSum = dfsSum(root.left, res);
            //res.add(leftSum);
            val += leftSum;
        }
        if (root.right != null) {
            int rightSum = dfsSum(root.right, res);
            //res.add(rightSum);
            val += rightSum;
        }
        res.add(val);
        return val;
    }

    public int[] findFrequentTreeSum(TreeNode root) {
        if (root == null)
            return new int[]{};
        List<Integer> res = new ArrayList<>();
        dfsSum(root, res);
        Map<Integer, Integer> cnt = new HashMap<>();
        int most = 0;
        for (int x : res) {
            if (!cnt.containsKey(x))
                cnt.put(x, 1);
            else
                cnt.put(x, cnt.get(x) + 1);
            most = Math.max(most, cnt.get(x));
        }
        res.clear();
        for (Map.Entry<Integer, Integer> entry : cnt.entrySet()) {
            if (entry.getValue() == most)
                res.add(entry.getKey());
        }

        int[] ans = new int[res.size()];
        for (int i = 0; i < ans.length; ++i)
            ans[i] = res.get(i);
        return ans;
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


    //leetcode 102 binary tree level order traversal
    //three ways of level order
    //recursive way to do level order;
    //O(n) time
    //O(h)space

    public void dfsLevelOrder(TreeNode root, List<List<Integer>> res, int level) {
        if (root == null)
            return;
        if (res.size() == level)
            res.add(new ArrayList<Integer>());
        res.get(level).add(root.val);
        if (root.left != null) {
            dfsLevelOrder(root.left, res, level + 1);
        }
        if (root.right != null) {
            //res.set(level,res.get(level)+" "+root.val);
            dfsLevelOrder(root.right, res, level + 1);
        }
    }

    public List<List<Integer>> levelOrderRecursive(TreeNode root) {
        List<List<Integer>> res = new ArrayList<List<Integer>>();
        if (root == null)
            return res;
        dfsLevelOrder(root, res, 0);
        return res;
    }

    //geeksforgeeks
    //Use function to print a given level
    //first you get the height of the tree
    //then print level one by one
    //time complexity is O(N2)
    public int getHeight(TreeNode root) {
        if (root == null)
            return 0;
        else return 1 + Math.max(getHeight(root.left), getHeight(root.right));
    }

    public void printGivenLevel(TreeNode root, int level) {
        if (root == null)
            return;
        if (level == 1)
            System.out.println(root.val + " ");
        else if (level > 1) {
            printGivenLevel(root.left, level - 1);
            printGivenLevel(root.right, level - 1);
        }
    }

    public void printLevelOrder(TreeNode node) {
        int h = getHeight(node);
        for (int i = 1; i <= h; ++i)
            printGivenLevel(node, i);
    }


    //iterative way to do level order;
    //time complexity is O(N),O(N) space

    public void levelOrder(TreeNode root) {
        if (root == null)
            return;
        Queue<TreeNode> q = new LinkedList<TreeNode>();
        q.offer(root);
        while (!q.isEmpty()) {
            int size = q.size();
            while (size-- > 0) {
                TreeNode node = q.poll();
                System.out.print(node.val + " ");
                if (node.left != null)
                    q.offer(node.left);
                if (node.right != null)
                    q.offer(node.right);
            }
            System.out.println();
        }
    }


    //three ways to traverse preorder,inorder,postorder
    public void preorder(TreeNode root){
        if(root==null)
            return;
        System.out.println(root.val);
        preorder(root.right);
        preorder(root.left);
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

    ////inorder traveral stack another version
    public void inorderByStack(TreeNode root) {
        Stack<TreeNode> stk = new Stack<TreeNode>();
        TreeNode cur = root;
        while (cur != null || !stk.isEmpty()) {
            while (cur != null) {
                stk.push(cur);
                cur = cur.left;
            }
            cur = stk.pop();
            System.out.println(cur.val);
            cur=cur.right;
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

    //postorder traveral stack another version

    public void postorderByStack(TreeNode root) {
        Stack<TreeNode> stk = new Stack<TreeNode>();
        TreeNode pre = null;
        TreeNode cur = root;
        while (cur != null || !stk.isEmpty()) {
            while (cur != null) {
                stk.push(cur);
                cur = cur.left;
            }
            cur = stk.peek();
            if (cur.right != null && pre != cur.right) {
                cur = cur.right;
            } else {//if there is no right child or right child has been visited, then you can start to visit the part
                System.out.println(cur.val);
                pre = cur;
                stk.pop();
                cur = null;
            }
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
