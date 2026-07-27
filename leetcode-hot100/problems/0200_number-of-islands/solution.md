# 200. 岛屿数量 - 题解

## 思路

遍历每个格子，遇到 '1' 就将整个连通分量通过 DFS/BFS 标记为 '0'（沉岛），岛屿计数 +1。

## 解法

```cpp
class Solution {
    void dfs(vector<vector<char>>& grid, int i, int j) {
        if (i < 0 || i >= grid.size() || j < 0 || j >= grid[0].size() || grid[i][j] != '1') return;
        grid[i][j] = '0';
        dfs(grid, i+1, j); dfs(grid, i-1, j); dfs(grid, i, j+1); dfs(grid, i, j-1);
    }
public:
    int numIslands(vector<vector<char>>& grid) {
        int count = 0;
        for (int i = 0; i < grid.size(); i++)
            for (int j = 0; j < grid[0].size(); j++)
                if (grid[i][j] == '1') { count++; dfs(grid, i, j); }
        return count;
    }
};
```

## 复杂度

- **时间**：O(m × n)
- **空间**：O(m × n)（递归栈，最坏全是陆地）

## 关键点

1. "沉岛"技巧：直接修改原数组标记已访问，无需额外 visited 数组
2. DFS/BFS/并查集三种方法都可以
