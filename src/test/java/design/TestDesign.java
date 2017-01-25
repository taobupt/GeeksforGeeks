package design;

/**
 * Created by Tao on 1/17/2017.
 */

import org.junit.Before;
import org.junit.Test;

import java.util.*;

import static org.junit.Assert.*;

public class TestDesign {
    MedianFinder md = null;
    Design dg = null;

    @Before
    public void setup() {
        md = new MedianFinder();
        dg = new Design();
    }

    @Test
    public void testMedia() {
        md.addNum(1);
        md.addNum(2);
        System.out.println(md.findMedian());// -> 1.5
        md.addNum(3);
        System.out.println(md.findMedian());// -> 2
        md.addNum(4);
        System.out.println(md.findMedian());
        md.addNum(5);
        System.out.println(md.findMedian());
    }

    @Test
    public void testWindowMedian() {
        int[] nums = {1, 3, -1, -3, 5, 3, 6, 7};
        dg.medianSlidingWindow(nums, 3);
    }

    @Test
    public void testTwitter() {
        Twitter tw = new Twitter();
        tw.postTweet(1, 5);
        tw.getNewsFeed(1);
        tw.follow(1, 2);
        tw.postTweet(2, 6);
        tw.getNewsFeed(1);
        tw.unfollow(1, 2);
        tw.getNewsFeed(1);
    }

    @Test
    public void testPhone() {
        PhoneDirectory pd = new PhoneDirectory(3);
        //["PhoneDirectory","check","get","get","release","check","get","get","check","get","check","check","get","release","check","check"]
        //[[3],[2],[],[],[2],[1],[],[],[0],[],[1],[2],[],[1],[1],[0]]
        pd.check(2);
        System.out.println(pd.get());
        System.out.println(pd.get());
        pd.release(2);
        System.out.println(pd.check(1));
        System.out.println(pd.get());
        System.out.println(pd.get());
        System.out.println(pd.check(0));
        System.out.println(pd.get());
        System.out.println(pd.check(1));
        System.out.println(pd.check(2));
        System.out.println(pd.get());
        pd.release(1);
        System.out.println(pd.check(1));
        System.out.println(pd.check(0));


    }

}
