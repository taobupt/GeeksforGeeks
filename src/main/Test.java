package main;

/**
 * Created by Tao on 12/22/2016.
 */
import LinkedList.*;
public class Test {

    public static void PrintList(ListNode head){
        while(head!=null){
            System.out.println(head.val+" ");
            head=head.next;
        }
    }
    public static void main(String []strs)
    {
        List l=new List();
        int []nums={1,2,3,4,5};
        l.createList(nums);


        int []nums1={1,2,3,4,5};
        List l1=new List();
        l1.createList(nums1);

        System.out.println("--------------------");
        Solution s=new Solution();
        //ListNode newNode=new ListNode(110);
        //PrintList(s.sortedInsert(l.getHead(),newNode));
        System.out.println(s.areIdentical(l1.getHead(),l.getHead()));
        s.swapPairs(l.getHead());


    }
}
