# 994. 腐烂的橘子 - 题解

## 思路

**多源 BFS**：所有腐烂橘子同时作为起点，一层一层向外扩散。

1. 将所有初始腐烂橘子入队，统计新鲜橘子数
2. BFS 按层扩散，每一层 = 1 分钟
3. 最后检查是否还有新鲜橘子

## 解法

```cpp
class Solution {
public:
    int orangesRotting(vector<vector<int>>& grid) {
        queue<pair<int,int>> q;
        int fresh = 0;
        for (...) { if (==2) q.push; if (==1) fresh++; }
        int minutes = 0;
        while (!q.empty()) { /* BFS 一层 */ minutes++; }
        return fresh == 0 ? minutes - 1 : -1;
    }
};
```

## 复杂度

- **时间**：O(m × n)
- **空间**：O(m × n)

## 关键点

1. **多源 BFS**：所有腐烂橘子同时入队，天然处理多个源
2. 结果是 `minutes - 1`（最后一层不再有新感染时多算了 1）
3. 如果初始就没有新鲜橘子，返回 0
