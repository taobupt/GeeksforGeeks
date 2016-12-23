package LinkedList;

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




}
