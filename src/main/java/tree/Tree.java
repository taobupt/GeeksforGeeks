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
                cur = cur.left;
                SUM += cur.val;
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
            TreeNode node = q.poll();
            System.out.print(node.val + " ");
            if (node.left != null)
                q.offer(node.left);
            if (node.right != null)
                q.offer(node.right);
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
