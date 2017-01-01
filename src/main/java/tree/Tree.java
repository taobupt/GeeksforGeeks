package tree;

import java.util.Stack;

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

    //you can solve this by previous node
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
