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
        int []nums={12,6,29};
        l.createList(nums);


        int []nums1={23,5,8};
        List l1=new List();
        l1.createList(nums1);

        int []nums2={90,20,59};
        List l2=new List();
        l2.createList(nums2);

        System.out.println("--------------------");
        Solution s=new Solution();
        //ListNode newNode=new ListNode(110);
        //PrintList(s.sortedInsert(l.getHead(),newNode));
        //s.AlternatingSplit(l.getHead());
        s.isSumsorted(l.getHead(),l1.getHead(),l2.getHead(),101);


    }
}
