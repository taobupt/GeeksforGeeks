package LinkedList;

/**
 * Created by Tao on 12/22/2016.
 */
public class List {
    private ListNode head;

    public List(){
        head=null;
    }

    public ListNode getHead(){
        return head;
    }

    public ListNode createList(int []nums){
        ListNode first=new ListNode(0);
        ListNode p=first;
        for(int x:nums){
            p.next=new ListNode(x);
            p=p.next;
        }
        head=first.next;
        return first.next;
    }

    public void push_front(int x){
        ListNode newNode=new ListNode(x);
        newNode.next=head;
        head=newNode;
    }

    public void append(int x){
        if(head==null){
            head=new ListNode(x);
            return;
        }
        ListNode p=head;
        while(p.next!=null){
            p=p.next;
        }
        p.next=new ListNode(x);
    }

    public void insertAfter(ListNode prev_node,int x){
        if(prev_node==null)return;
        ListNode newNode=new ListNode(x);
        newNode.next=prev_node.next;
        prev_node.next=newNode;
    }

    public void printList(){
        ListNode p=head;
        while(p!=null){
            System.out.println(p.val);
            p=p.next;
        }
    }

    public void deleteNode(int pos ){
        ListNode first=new ListNode(0);
        ListNode p=first;
        first.next=head;
        while(pos-->0){
            p=p.next;
        }
        p.next=p.next.next;
        head=first.next;
    }

    public ListNode deleteDuplicates(ListNode head) {
        if(head==null||head.next==null)return head;
        ListNode p=head;
        ListNode q=head.next;
        ListNode qparent=head;
        while(qparent!=null && q!=null)
        {
            if(q.val!=qparent.val){
                p.next=q;
                p=p.next;
            }
            qparent=qparent.next;
            q=q.next;
        }
        p.next=null;//do not forget this
        return head;
    }

    //more optimal
    public ListNode deleteDuplicateOptimal(ListNode head){
        if(head==null||head.next==null)return head;
        ListNode p=head;
        while(p.next!=null){
            if(p.next.val==p.val)p.next=p.next.next;
            else p=p.next;
        }
        return head;
    }
    //Recursive way
    public ListNode deleteDuplicateRecursive(ListNode head){
        if(head==null ||head.next==null)return head;
        head.next=deleteDuplicateRecursive(head.next);
        return head.val==head.next.val?head.next:head;
    }

}
