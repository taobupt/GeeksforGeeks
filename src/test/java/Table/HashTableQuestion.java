package Table;

/**
 * Created by tao on 1/24/17.
 */

import hashTable.HASHTABLE;
import org.junit.Before;
import org.junit.Test;

import java.beans.Transient;
import java.util.*;

import static org.junit.Assert.*;

public class HashTableQuestion {
    HASHTABLE ht = null;

    @Before
    public void setup() {
        ht = new HASHTABLE();
    }

    @Test
    public void testFindStrobogrammatic() {
        ht.findStrobogrammatic(4);
    }
}
