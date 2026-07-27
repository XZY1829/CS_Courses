# 51. N 皇后 - 题解

## 思路

回溯：逐行放置皇后，用三个布尔数组标记被攻击的列、主对角线（`row-col`）、副对角线（`row+col`）。

## 解法

```cpp
class Solution {
    vector<vector<string>> result;
    vector<bool> cols, diag1, diag2;
    void backtrack(int n, int row, vector<string>& board) {
        if (row == n) { result.push_back(board); return; }
        for (int col = 0; col < n; col++) {
            if (cols[col] || diag1[row-col+n] || diag2[row+col]) continue;
            board[row][col] = 'Q';
            cols[col] = diag1[row-col+n] = diag2[row+col] = true;
            backtrack(n, row+1, board);
            board[row][col] = '.';
            cols[col] = diag1[row-col+n] = diag2[row+col] = false;
        }
    }
};
```

## 复杂度

- **时间**：O(n!)
- **空间**：O(n)

## 关键点

1. 主对角线上 `row - col` 相同，副对角线上 `row + col` 相同
2. `row - col` 可能为负，加 n 偏移避免负索引
3. 逐行放置保证每行恰好一个皇后，只需检查列和对角线冲突
