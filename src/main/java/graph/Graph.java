package graph;

import java.util.*;

/**
 * Created by Tao on 2/5/2017.
 */

public class Graph {

    //399 evaluate division
    public double[] calcEquation(String[][] equations, double[] values, String[][] queries) {
        Map<String, ArrayList<String>> pairs = new HashMap<>();
        Map<String, ArrayList<Double>> valuePair = new HashMap<>();
        for (int i = 0; i < equations.length; ++i) {
            String[] equation = equations[i];
            if (!pairs.containsKey(equation[0])) {
                pairs.put(equation[0], new ArrayList<String>());
                valuePair.put(equation[0], new ArrayList<Double>());
            }
            if (!pairs.containsKey(equation[1])) {
                pairs.put(equation[1], new ArrayList<String>());
                valuePair.put(equation[1], new ArrayList<Double>());
            }
            pairs.get(equation[0]).add(equation[1]);
            pairs.get(equation[1]).add(equation[0]);
            valuePair.get(equation[0]).add(values[i]);
            valuePair.get(equation[1]).add(1.0 / values[i]);
        }
        double[] result = new double[queries.length];
        for (int i = 0; i < queries.length; ++i) {
            String[] query = queries[i];
            result[i] = dfs(query[0], query[1], pairs, valuePair, new HashSet<String>(), 1.0);
            if (result[i] == 0.0)
                result[i] = -1.0;
        }
        return result;
    }

    public double dfs(String start, String end, Map<String, ArrayList<String>> pairs, Map<String, ArrayList<Double>> values, Set<String> set, double value) {
        if (set.contains(start))
            return 0.0;
        if (!pairs.containsKey(start))
            return 0.0;
        if (start.equals(end))
            return value;
        set.add(start);
        List<String> strList = pairs.get(start);
        List<Double> valueList = values.get(start);
        double tmp = 0.0;
        for (int i = 0; i < strList.size(); ++i) {
            tmp = dfs(strList.get(i), end, pairs, values, set, value * valueList.get(i));
            if (tmp != 0.0)
                break;
        }
        set.remove(start);
        return tmp;
    }
}
