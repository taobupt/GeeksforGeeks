package design;

import java.util.*;

/**
 * Created by Tao on 1/17/2017.
 */

class TicTacToe {
    private int[][] rows;
    private int[][] cols;
    private int[] diag = new int[2];
    private int[] adiag = new int[2];
    private int size;

    public TicTacToe(int n) {
        size = n;
        rows = new int[n][2];
        cols = new int[n][2];
    }

    public int move(int row, int col, int player) {
        rows[row][player - 1]++;//  add number
        if (rows[row][player - 1] == size)
            return player;
        cols[col][player - 1]++;
        if (cols[col][player - 1] == size)
            return player;
        if (row == col) {
            diag[player - 1]++;
            if (diag[player - 1] == size)
                return player;
        }
        if (row + col == size - 1) {
            adiag[player - 1]++;
            if (adiag[player - 1] == size)
                return player;
        }
        return 0;
    }
}

class HitCounter {

    /**
     * Initialize your data structure here.
     */
    private Deque<Integer> dq = null;

    public HitCounter() {
        dq = new LinkedList();
    }

    /**
     * Record a hit.
     *
     * @param timestamp - The current timestamp (in seconds granularity).
     */
    public void hit(int timestamp) {
        dq.addLast(timestamp);
        while (!dq.isEmpty() && dq.peekFirst() <= timestamp - 300) {
            dq.pollFirst();
        }
    }

    /**
     * Return the number of hits in the past 5 minutes.
     *
     * @param timestamp - The current timestamp (in seconds granularity).
     */
    public int getHits(int timestamp) {
        return dq.size();
    }
}

//Find Median from Data Stream

class MedianFinder {
    PriorityQueue<Integer> pq1 = null;//large elements
    PriorityQueue<Integer> pq2 = null;//small elements
    // Adds a number into the data structure.

    public MedianFinder() {
        pq2 = new PriorityQueue<>();
        pq1 = new PriorityQueue<>(Collections.reverseOrder());
    }

    public void addNum(int num) {
        while (pq2.size() < pq1.size()) {
            pq2.add(pq1.poll());
        }
        if (pq2.isEmpty() || num >= pq2.peek())
            pq2.add(num);
        else
            pq1.add(num);
        while (pq2.size() < pq1.size()) {
            pq2.add(pq1.poll());
        }
        while (pq2.size() > pq1.size() + 1) {
            pq1.add(pq2.poll());
        }
    }

    // Returns the median of current data stream
    public double findMedian() {
        int size = pq1.size() + pq2.size();
        return size % 2 == 0 ? (pq1.peek() / 2.0 + pq2.peek() / 2.0) : pq2.peek();
    }

};

class Tuple<X, Y> {
    public final X x;
    public final Y y;

    public Tuple(X x, Y y) {
        this.x = x;
        this.y = y;
    }
}

class Twitter {

    int cnt = 0;
    private Map<Integer, List<Integer>> relation;
    private Map<Integer, PriorityQueue<Tuple<Integer, Integer>>> tweet;

    /**
     * Initialize your data structure here.
     */
    public Twitter() {
        relation = new HashMap<>();
        tweet = new HashMap<>();
    }

    /**
     * Compose a new tweet.
     */
    public void postTweet(int userId, int tweetId) {
        if (tweet.containsKey(userId)) {
            tweet.get(userId).add(new Tuple(tweetId, cnt++));
            if (tweet.get(userId).size() > 10) {
                List<Tuple<Integer, Integer>> save = new ArrayList<>();
                while (tweet.get(userId).size() > 1) {
                    save.add(tweet.get(userId).poll());
                }
                tweet.get(userId).poll();
                tweet.get(userId).addAll(save);
            }
        } else {
            PriorityQueue<Tuple<Integer, Integer>> tmp = new PriorityQueue<>(new Comparator<Tuple<Integer, Integer>>() {
                @Override
                public int compare(Tuple<Integer, Integer> o1, Tuple<Integer, Integer> o2) {
                    if (o1.y > o2.y)
                        return -1;
                    else if (o1.y < o2.y)
                        return 1;
                    else
                        return 0;
                }
            });
            tmp.add(new Tuple(tweetId, cnt++));
            tweet.put(userId, tmp);
        }
    }

    /**
     * Retrieve the 10 most recent tweet ids in the user's news feed. Each item in the news feed must be posted by users who the user followed or by the user herself. Tweets must be ordered from most recent to least recent.
     */
    public List<Integer> getNewsFeed(int userId) {
        List<Integer> res = new ArrayList<>();
        if (!relation.containsKey(userId)) {
            if (tweet.containsKey(userId)) {
                PriorityQueue<Tuple<Integer, Integer>> tmp = new PriorityQueue<>(tweet.get(userId));
                while (!tmp.isEmpty()) {
                    res.add(tmp.poll().x);
                }
            }
        } else {
            PriorityQueue<Tuple<Integer, Integer>> pq = new PriorityQueue<>(new Comparator<Tuple<Integer, Integer>>() {
                @Override
                public int compare(Tuple<Integer, Integer> o1, Tuple<Integer, Integer> o2) {
                    if (o1.y > o2.y)
                        return -1;
                    else if (o1.y < o2.y)
                        return 1;
                    else
                        return 0;
                }
            });
            if (tweet.containsKey(userId))
                pq.addAll(tweet.get(userId));
            for (int id : relation.get(userId)) {
                if (tweet.containsKey(id))
                    pq.addAll(tweet.get(id));
            }
            int i = 0;
            while (i < 10 && !pq.isEmpty()) {
                res.add(pq.poll().x);
                i++;
            }
        }
        return res;
    }

    /**
     * Follower follows a followee. If the operation is invalid, it should be a no-op.
     */
    public void follow(int followerId, int followeeId) {
        if (relation.containsKey(followerId)) {
            relation.get(followerId).add(followeeId);
        } else {
            List<Integer> tmp = new ArrayList<>();
            tmp.add(followeeId);
            relation.put(followerId, tmp);
        }
    }

    /**
     * Follower unfollows a followee. If the operation is invalid, it should be a no-op.
     */
    public void unfollow(int followerId, int followeeId) {
        if (relation.containsKey(followerId)) {
            if (relation.get(followerId).contains(followeeId)) {
                if (relation.get(followerId).size() == 1)
                    relation.remove(followerId);
                else {
                    int idx = relation.get(followerId).indexOf(followeeId);
                    relation.get(followerId).remove(idx);
                }
            }
        }
    }
}

public class Design {
    //480 sliding window median
    public double[] medianSlidingWindow(int[] nums, int k) {
        int begin = 0, n = nums.length;
        MedianFinder md = new MedianFinder();
        double[] res = new double[n - k + 1];
        while (begin < n) {
            if (begin < k) {
                md.addNum(nums[begin]);
            } else {
                res[begin - k] = md.findMedian();
                if (nums[begin - k] >= res[begin - k]) {
                    md.pq2.remove(nums[begin - k]);
                } else
                    md.pq1.remove(nums[begin - k]);
                md.addNum(nums[begin]);
            }
            begin++;
        }
        res[n - k] = md.findMedian();
        return res;
    }
}
