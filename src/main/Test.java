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
        int []nums={9,9,9};
        l.createList(nums);


        int []nums1={2,3,20};
        List l1=new List();
        l1.createList(nums1);

        int []nums2={8,4,2};
        List l2=new List();
        l2.createList(nums2);

        System.out.println("--------------------");
        Solution s=new Solution();
        //ListNode newNode=new ListNode(110);
        //PrintList(s.sortedInsert(l.getHead(),newNode));
        //s.AlternatingSplit(l.getHead());
        //PrintList(s.addTwoListRecursive(l.getHead(),l1.getHead()));
        //PrintList(s.addTwoLists2(l.getHead(),l1.getHead()));
        PrintList(s.plusOne(l.getHead()));


    }
}
