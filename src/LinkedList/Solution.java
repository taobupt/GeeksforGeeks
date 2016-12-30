package LinkedList;

import java.util.*;

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

    //merge two sortedlist in reverseorder
    public ListNode mergeTwoSortedListReverse(ListNode heada,ListNode headb){
        ListNode dummy=mergeTwoLists(heada,headb);
        return reverseList(dummy);
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
    //delete a linkedlist node at given position

    public ListNode deleteNode(ListNode head,int position){
        if(head==null)return head;
        ListNode dummy=new ListNode(0);
        dummy.next=head;
        ListNode p=dummy;
        while(p!=null && position>0){
            p=p.next;
            position--;
        }
        if(p!=null &&p.next!=null)p.next=p.next.next;
        return dummy.next;
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


    // practice questions for linked list and recursion
    //print node recursively

    public void fun1(ListNode head){
        if(head==null)
            return;
        fun1(head.next);
        System.out.println(head.val+" ");
    }

    //just print the odd node
    //pay attention to the odd length and even length
    public void fun2(ListNode head){
        if(head==null)return;
        System.out.println(head.val+" ");

        if(head.next!=null){
            fun2(head.next.next);
        }
        System.out.println(head.val+" ");
    }

    //move last element to front of a given linkedlist

    public ListNode moveToFront(ListNode head){
        if(head==null||head.next==null)return head;
        ListNode p=head;
        ListNode pparent=new ListNode(0);
        pparent.next=p;
        while(p.next!=null){
            p=p.next;
            pparent=pparent.next;
        }
        pparent.next=null;
        p.next=head;
        return p;
    }

    //intesection of two sorted linkedlist
    //iterative way
    public ListNode sortedIntersect(ListNode heada,ListNode headb){
        ListNode dummy=new ListNode(0);
        ListNode p=dummy;
        ListNode nodea=heada;
        ListNode nodeb=headb;
        while(nodea!=null && nodeb!=null){
            if(nodea.val<nodeb.val)
                nodea=nodea.next;
            else if(nodeb.val<nodea.val)
                nodeb=nodeb.next;
            else{
                p.next=nodea;
                p=p.next;
                nodea=nodea.next;
                nodeb=nodeb.next;
            }
        }
        p.next=null;
        return dummy.next;
    }

    //interesting
    //recursive
    public ListNode sortedIntersectRecursive(ListNode heada,ListNode headb){
        if(heada==null||headb==null)return null;
        if(heada.val<headb.val)
            return sortedIntersectRecursive(heada.next,headb);
        else if(heada.val>headb.val)
            return sortedIntersectRecursive(heada,headb.next);
        else{
            ListNode node=new ListNode(heada.val);
            node.next=sortedIntersectRecursive(heada.next,headb.next);
            return node;
        }
    }

    //delete alternate nodes of a linked list
    //recursive

    public ListNode deleteAlRecursive(ListNode head){
        if(head==null||head.next==null)return head;
        head.next=deleteAlRecursive(head.next.next);
        return head;
    }

    //iterative way

    public ListNode deleteAl(ListNode head){
        if(head==null||head.next==null)return head;
        ListNode node=head;
        while(node!=null &&node.next!=null){
            node.next=node.next.next;
            node=node.next;
        }
        return head;
    }

    //alternating split of a given singly linked list
    public void AlternatingSplit(ListNode head){
        ListNode list1=new ListNode(0);
        ListNode list2=new ListNode(0);
        int count=0;
        ListNode p1=list1;
        ListNode p2=list2;
        ListNode p=head;
        while(p!=null){
            if((count&0x1)==0){
                p1.next=p;
                p1=p1.next;
                p=p.next;
            }else{
                p2.next=p;
                p2=p2.next;
                p=p.next;
            }
            count++;
        }
        //pay attention to this
        if(p1!=null)p1.next=null;
        if(p2!=null)p2.next=null;
        while(list1!=null){
            System.out.println(list1.val+" ");
            list1=list1.next;
        }

        while(list2!=null){
            System.out.println(list2.val+" ");
            list2=list2.next;
        }

    }

    //two sum
    public int[] twoSum(int[] nums, int target) {
        int[]result=new int[2];
        Map<Integer,Integer>map=new HashMap<Integer, Integer>();
        for(int i=0;i<nums.length;++i){
            if(map.containsKey(target-nums[i])){
                result[1]=i;
                result[0]=map.get(target-nums[i]);
                return result;
            }
            map.put(nums[i],i);
        }
        return result;
    }

    //rotate linkedlist
    //3 steps
    //or you can reverse once, find the point,and then connect two parts directly,
   public ListNode rotate(ListNode head,int k){
        if(head==null||k==0)return head;
        int n=0;
        ListNode node=head;
        while(node!=null){
            node=node.next;
            n++;
        }
        k=k%n;
        if(k==0)return head;
        node=head;
        while(k-->1){
            node=node.next;
        }
        ListNode second=node.next;
        node.next=null;
        second=reverseList(second);
        node=head;
        node=reverseList(node);
        head.next=second;
        head=reverseList(node);
        return head;
    }

    public ListNode removeDuplicates(ListNode head){
        if(head==null||head.next==null)return head;
        Set<Integer>set=new HashSet<Integer>();
        ListNode front=head;
        ListNode node=head;
        set.add(front.val);
        while(node!=null){
            if(set.contains(node.val)){
                front.next=node.next;
            }else
                front=front.next;
            set.add(node.val);
            node=node.next;

        }
        return head;
    }



    //5-6-3// 365
    public ListNode addTwoLists(ListNode heada,ListNode headb){
        if(heada==null||headb==null)return heada==null?headb:heada;
        int carry=0;
        ListNode first=new ListNode(0);
        ListNode pp=first;
        while(heada!=null||headb!=null||carry>0){
            carry+=(heada!=null?heada.val:0)+(headb!=null?headb.val:0);
            pp.next=new ListNode(carry%10);
            pp=pp.next;
            carry/=10;
            heada=heada!=null?heada.next:null;
            headb=headb!=null?headb.next:null;
        }
        return first.next;
    }

    //add tow list
    //can also use two stack
    //5-6-3// 563
    //itreative way
    public ListNode addTwoLists2(ListNode heada,ListNode headb){
        if(heada==null||headb==null)return heada==null?headb:heada;
        heada=reverseList(heada);
        headb=reverseList(headb);
        return reverseList(addTwoLists(heada,headb));
    }

    //recursive way
    public ListNode addTwoListRecursive(ListNode heada,ListNode headb){
        return addTwoListRecursiveHelper(heada,headb,0);
    }

    public ListNode addTwoListRecursiveHelper(ListNode heada,ListNode headb,int carry){
        if(heada==null && headb==null && carry==0)
            return null;
        else if(heada==null && headb==null && carry!=0)
            return new ListNode(carry);
        int firstData=heada==null?0:heada.val;
        int secondData=headb==null?0:headb.val;
        ListNode sum=new ListNode((firstData+secondData+carry)%10);
        int newCarray=(firstData+secondData+carry)/10;
        heada=heada==null?null:heada.next;
        headb=headb==null?null:headb.next;
        sum.next=addTwoListRecursiveHelper(heada,headb,newCarray);
        return sum;
    }


    public ListNode segregateEvenOdd(ListNode head){
        ListNode even=new ListNode(0);
        ListNode peven=even;
        ListNode odd=new ListNode(0);
        ListNode podd=odd;

        ListNode node=head;
        while(node!=null){
            if((node.val&0x1)==0){
                peven.next=node;
                peven=peven.next;
            }else{
                podd.next=node;
                podd=podd.next;
            }
            node=node.next;
        }
        peven.next=odd.next;
        podd.next=null;
        return even.next;
    }
    //Delete nodes which have a greater value on right side
    //interesting
    //Method 2 (Use Reverse)
    //1. Reverse the list.
    //2. Traverse the reversed list. Keep max till now. If next node < max, then delete the next node, otherwise max = next node.
    //3. Reverse the list again to retain the original order. Time Complexity: O(n) Thanks to R.Srinivasan for providing below code.



    //merge sort list

    public enum sortType{desc,asc};
    public ListNode merge(ListNode heada,ListNode headb,sortType type){
        if(heada==null||headb==null)return heada!=null?heada:headb;
        ListNode headc=new ListNode(0);
        ListNode p=headc;
        while(heada!=null && headb!=null){
            if(type==sortType.desc &&heada.val>=headb.val||type==sortType.asc && heada.val<=headb.val){
                p.next=heada;
                heada=heada.next;
            }else{
                p.next=headb;
                headb=headb.next;
            }
            p=p.next;
        }
        p.next=heada!=null?heada:headb;
        return headc.next;
    }
    public ListNode mergeSort(ListNode head,sortType type){
        if(head==null||head.next==null)return head;
        ListNode slow=head;
        ListNode fast=head.next;
        while(fast!=null && fast.next!=null){
            fast=fast.next.next;
            slow=slow.next;
        }
        fast=slow.next;
        slow.next=null;
        return merge(mergeSort(head,type),mergeSort(fast,type),type);
    }


    //find a triplet from three linked lists with sum equal to a given number
    //first brust force
    //second sort b in ascend ,sort c in descend

    public boolean isSumsorted(ListNode heada,ListNode headb,ListNode headc,int givenNumber){
        ListNode nodea=heada;
        headb=mergeSort(headb,sortType.asc);
        headc=mergeSort(headc,sortType.desc);
        while(nodea!=null){
            ListNode b=headb;
            ListNode c=headc;
            while(b!=null && c!=null){
                int sum=nodea.val+b.val+c.val;
                if(sum==givenNumber){
                    System.out.println("Triplet found "+nodea.val+" "+b.val+" "+c.val);
                    return true;
                }else if(sum<givenNumber)
                    b=b.next;
                else
                    c=c.next;
            }
            nodea=nodea.next;
        }
        System.out.println("No Triplet found");
        return false;
    }

    //delete last occurence of an item from linked list
    public ListNode deleteLast(ListNode head,int key){
        if(head==null)return head;
        ListNode dummy=new ListNode(0);
        ListNode p=dummy;
        dummy.next=head;
        ListNode node=null;
        while(dummy.next!=null){
            if(dummy.next.val==key){
                node=dummy;
            }
            dummy=dummy.next;
        }
        if(node!=null)node.next=node.next.next;
        return p.next;
    }

    //without reverselist
    public ListNode plusOne(ListNode head) {
        ListNode dummy = new ListNode(0);
        dummy.next = head;
        ListNode i = dummy;
        ListNode j = dummy;

        while (j.next != null) {
            j = j.next;
            if (j.val != 9) {
                i = j;
            }
        }

        if (j.val != 9) {
            j.val++;
        } else {
            i.val++;
            i = i.next;
            while (i != null) {
                i.val = 0;
                i = i.next;
            }
        }

        if (dummy.val == 0) {
            return dummy.next;
        }

        return dummy;
    }
    //using reverselist
    public ListNode plusOne1(ListNode head) {
        if(head==null)return new ListNode(1);
        head=reverseList(head);
        ListNode p=head;
        ListNode pre=new ListNode(0);
        pre.next=p;
        while(p!=null){
            if(p.val!=9){
                p.val++;
                return reverseList(head);
            }else p.val=0;
            p=p.next;
            pre=pre.next;
        }
        pre.next=new ListNode(1);
        return reverseList(head);
    }

    //is palindrome of string of linkedlist

    public boolean isPalindromeof(ListNode head){
        StringBuffer sb=new StringBuffer("");
        while(head!=null){
            sb.append(String.valueOf(head.val));
        }
        int n=sb.length();
        for(int i=0;i<n;++i) {
            if (sb.charAt(i) != sb.charAt(n - 1 - i)) return false;
        }
        return true;
    }

    //judge whether string is a palindrome
    public boolean isPalindrome(String s) {
        int i=0,j=s.length()-1;
        while(i<j)
        {
            while(i<j && !Character.isLetterOrDigit(s.charAt(j)))j--;
            while(i<j && !Character.isLetterOrDigit(s.charAt(i)))i++;
            if(Character.toLowerCase(s.charAt(i++))!=Character.toLowerCase(s.charAt(j--)))return false;
        }
        return true;
    }

}
