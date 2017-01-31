package common;

/**
 * Created by Tao on 1/30/2017.
 */
public class Interval {
    public int start;
    public int end;

    public Interval(int s, int e) {
        this.start = s;
        this.end = e;
    }

    public Interval() {
        this.start = 0;
        this.end = 0;
    }

    public String toString() {
        return "[" + start + " " + end + "]";
    }
}
