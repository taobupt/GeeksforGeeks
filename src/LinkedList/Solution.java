package LinkedList;

import java.util.Stack;

/**
 * Created by Tao on 12/22/2016.
 */
public class Solution {
    //get length
    //iterative
    public int getLength(ListNode head){
        int count=0;
        ListNode p=head;
        while(p!=null){
            count++;
            p=p.next;
        }
        return count;
    }

    //recursive way
    public int getLengthRecursive(ListNode head){
        if(head==null)return 0;
        else return 1+getLengthRecursive(head.next);
    }

    //iterative way
    public boolean search(ListNode head,int val){
        ListNode p=head;
        while(p!=null){
            if(p.val==val)return true;
            p=p.next;
        }
        return false;
    }

    //recursive way
    public boolean searchRecursive(ListNode head,int val){
        if(head==null)return false;
        if(head.val==val)return true;
        else return searchRecursive(head.next,val);
    }

    //swap nodes in a linked list without swapping data

    public ListNode swapNodes(ListNode head,int x,int y){
        ListNode first=new ListNode(0);
        first.next=head;
        ListNode nodex=first;
        ListNode nodey=first;
        ListNode savex=null;
        ListNode savey=null;
        ListNode p=first;
        while(p.next!=null){
            if(p.next.val==x){
                savex=p.next;
                nodex=p;
            }else if(p.next.val==y){
                savey=p.next;
                nodey=p;
            }
            p=p.next;
        }
        if(savex!=null && savey!=null){
            nodex.next=savey;
            nodey.next=savex;

            //swap next pointers
            ListNode tmp=savex.next;
            savex.next=savey.next;
            savey.next=tmp;
        }
        return first.next;
    }

    //iterative way
    public int getNth(ListNode head,int index){
        ListNode p=head;
        while(index-->0){
            if(p!=null)p=p.next;
        }
        return p!=null?p.val:Integer.MIN_VALUE;
    }

    //recursive way

    public int getNthRecursive(ListNode head,int index){
        if(head==null)return Integer.MIN_VALUE;
        if(index==0)return head.val;
        return getNthRecursive(head.next,index-1);
    }

    //iterative way
    public ListNode reverseList(ListNode head){
        ListNode pre=null;
        while(head!=null){
            ListNode tmp=head.next;
            head.next=pre;
            pre=head;
            head=tmp;
        }
        return pre;
    }

    //recursive way
    public ListNode reverseListRecursive(ListNode head){
        if(head==null||head.next==null)return head;
        ListNode next=reverseListRecursive(head.next);
        head.next.next=head;
        head.next=null;
        return next;
    }

    //find the middle of a given linked list
    // two pointers is the better
    // if you want get the first middle, you should set fast=head.next;
    public void printMiddle(ListNode head){
        if(head==null)return;
        ListNode fast=head;
        ListNode slow=head;
        while(fast!=null && fast.next!=null){
            fast=fast.next.next;
            slow=slow.next;
        }
        System.out.println(slow.val);
    }

    public void printNthFromLast(ListNode head, int n) {
        ListNode p=head;
        ListNode q=head;
        while(n-->0){
            if(p!=null)p=p.next;
            else
            {
                System.out.println("exceed the length");
                return;
            }
        }
        while(p!=null){
            p=p.next;
            q=q.next;
        }
        System.out.println(q.val);
    }


    //detect loop
    public boolean hasCycle(ListNode head) {
        if(head==null||head.next==null)return false;
        ListNode fast=head;
        ListNode slow=head;
        while(fast!=null && fast.next!=null){
            fast=fast.next.next;
            slow=slow.next;
            if(fast==slow)return true;
        }
        return false;

    }

    //merge two sorted list
    //iterative way
    public ListNode mergeTwoLists(ListNode l1, ListNode l2) {
        ListNode first=new  ListNode(0);
        ListNode p=first;
        if(l1==null||l2==null)return l1!=null?l1:l2;
        while(l1!=null && l2!=null){
            if(l1.val>=l2.val){
                p.next=l2;
                l2=l2.next;
            }else{
                p.next=l1;
                l1=l1.next;
            }
            p=p.next;
        }
        p.next=l1!=null?l1:l2;
        return first.next;
    }

    //recursive way

    public ListNode mergeTwoListsRecursive(ListNode l1,ListNode l2){
        if(l1==null||l2==null)return l1!=null?l1:l2;
        if(l1.val<l2.val){
            l1.next=mergeTwoListsRecursive(l1.next,l2);
            return l1;
        }else{
            l2.next=mergeTwoListsRecursive(l1,l2.next);
            return l2;
        }
    }

    public ListNode sortedInsert(ListNode head,ListNode newNode){
        ListNode dummy=new ListNode(Integer.MIN_VALUE);
        dummy.next=head;
        ListNode p=dummy;
        while(p.next!=null){
            if(p.next.val>=newNode.val){
                break;
            }
            p=p.next;
        }
        newNode.next=p.next;
        p.next=newNode;
        return dummy.next;
    }

    //leetcode 234 palindrome linked list



    public boolean isPalindrome(ListNode head){
        if(head==null||head.next==null)return true;
        ListNode slow=head;
        ListNode fast=head.next;
        while(fast!=null && fast.next!=null){
            slow=slow.next;
            fast=fast.next.next;
        }
        ListNode second=slow.next;
        slow.next=null;
        slow=head;
        second=reverseList(second);
        while(second!=null){
            if(slow.val!=second.val)return false;
            slow=slow.next;
            second=second.next;
        }
        return true;
    }

    ListNode cur;
    public boolean isPalindromeRecursive(ListNode head){
        if(head==null||head.next==null)return true;
        cur=head;
        return isPalindromeRecursiveHelper(head);
    }

    public boolean isPalindromeRecursiveHelper(ListNode head){
        if(head==null)return true;
        boolean flag=isPalindromeRecursiveHelper(head.next);
        if(flag && head.val==cur.val){
            cur=cur.next;
            return true;
        }
        return false;
    }

    //print reversely
    //recursive way
    public void printReverseRecursive(ListNode head){
        if(head==null)return;
        printReverseRecursive(head.next);
        System.out.println(head.val+" ");
    }

    //iterative way
    //you can use a stack or reverselist first
    public void printReverse(ListNode head){
        if(head==null)return;
        head=reverseList(head);
        ListNode node=head;
        while(node!=null){
            System.out.println(node.val+" ");
            node=node.next;
        }
    }


    public void printReverseByStack(ListNode head){
        if(head==null)return;
        Stack<ListNode> stk=new Stack<>();
        ListNode node=head;
        while(node!=null){
            stk.push(node);
            node=node.next;
        }
        while(!stk.isEmpty())
        {
            node=stk.pop();
            System.out.println(node.val+" ");
        }
    }

    //just swap node val, and then delete node
    public void deleteNode(ListNode node) {
        if(node==null)return;
        //int tmp=node.val;
        node.val=node.next.val;
        node.next=node.next.next;
    }


    //iterative way
    public boolean areIdentical(ListNode lista,ListNode listb){
        while(lista!=null && listb!=null){
            if(lista.val!=listb.val)return false;
            lista=lista.next;
            listb=listb.next;
        }
        return (lista==null && listb==null);
    }

    //recursive way

    public boolean areIdenticalRecursive(ListNode lista,ListNode listb){
        if(lista==null||listb==null)return lista==listb;
        if(lista.val==listb.val && areIdenticalRecursive(lista.next,listb.next))
            return true;
        return false;
    }

    //swap nodes in pairs
    //recursive way

    public ListNode swapPairsRecursive(ListNode head){
        if(head==null||head.next==null)return head;
        ListNode node=head.next;
        head.next=swapPairsRecursive(head.next.next);
        node.next=head;
        return node;
    }

    //iterative way;
    //split two list and merge
    public ListNode swapPairs(ListNode head){
        if(head==null||head.next==null)return head;
        ListNode odd=head;
        ListNode even=head.next;
        ListNode saveodd=odd;
        ListNode saveeven=even;
        int count=2;
        ListNode node=even.next;
        while(node!=null){
            if((count&1)==0){
                odd.next=node;
                odd=odd.next;
            }else
            {
                even.next=node;
                even=even.next;
            }
            node=node.next;
            count++;
        }
        even.next=null;
        odd.next=null;
        even=saveeven;
        odd=saveodd;
        ListNode dummy=new ListNode(0);
        node=dummy;
        count=0;
        while(odd!=null && even!=null){
            if((count&0x1)==0){
                node.next=even;
                even=even.next;
            }else{
                node.next=odd;
                odd=odd.next;
            }
            node=node.next;
        }
        node.next=odd;
        count++;
        return dummy;

    }







}
