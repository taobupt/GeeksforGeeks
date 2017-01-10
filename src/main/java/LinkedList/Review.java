package LinkedList;

/**
 * Created by Tao on 1/9/2017.
 */
public class Review {
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
        return dummy.val == 0 ? dummy.next : dummy;
    }
}
