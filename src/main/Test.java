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
        l.printList();

        System.out.println("--------------------");
        Solution s=new Solution();
        PrintList(s.swapNodes(l.getHead(),3,4));

        //PrintList(s.reverseList(l.getHead()));
        //PrintList(s.reverseListRecursive(l.getHead()));
        //s.printNthFromLast(l.getHead(),1);

    }
}
