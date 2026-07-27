# 79. 单词搜索 - 题解

## 思路

DFS + 回溯：从每个格子开始，四方向搜索，用临时标记 `'#'` 防止重复使用同一格子。

## 解法

```cpp
class Solution {
    bool dfs(vector<vector<char>>& board, const string& word, int i, int j, int k) {
        if (k == word.size()) return true;
        if (i<0 || i>=board.size() || j<0 || j>=board[0].size() || board[i][j]!=word[k]) return false;
        char tmp = board[i][j];
        board[i][j] = '#';
        bool found = dfs(...i+1...) || dfs(...i-1...) || dfs(...j+1...) || dfs(...j-1...);
        board[i][j] = tmp;
        return found;
    }
};
```

## 复杂度

- **时间**：O(m × n × 3^L)，L 为单词长度
- **空间**：O(L)

## 关键点

1. 用原地标记 `'#'` 代替 visited 数组
2. 回溯时恢复原字符
3. 3^L 而非 4^L，因为不会回到来时方向
