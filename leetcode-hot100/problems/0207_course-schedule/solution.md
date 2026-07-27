# 207. 课程表 - 题解

## 思路

判断有向图是否有环 = **拓扑排序**（Kahn 算法 / BFS）。

1. 建图 + 统计入度
2. 入度为 0 的节点入队
3. BFS：每次取出入度 0 的节点，将其邻居入度 -1，若变成 0 则入队
4. 如果最终处理的节点数 = 总课程数，则无环

## 解法

```cpp
class Solution {
public:
    bool canFinish(int numCourses, vector<vector<int>>& prerequisites) {
        vector<vector<int>> graph(numCourses);
        vector<int> inDegree(numCourses, 0);
        for (auto& p : prerequisites) { graph[p[1]].push_back(p[0]); inDegree[p[0]]++; }
        queue<int> q;
        for (int i = 0; i < numCourses; i++) if (inDegree[i] == 0) q.push(i);
        int count = 0;
        while (!q.empty()) {
            int cur = q.front(); q.pop(); count++;
            for (int next : graph[cur]) if (--inDegree[next] == 0) q.push(next);
        }
        return count == numCourses;
    }
};
```

## 复杂度

- **时间**：O(V + E)
- **空间**：O(V + E)

## 关键点

1. 拓扑排序是有向无环图（DAG）的标准工具
2. 也可以用 DFS + 三色标记法检测环
