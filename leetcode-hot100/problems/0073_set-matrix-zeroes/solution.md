# 73. 矩阵置零 - 题解

## 思路

O(m+n) 空间很简单（记录哪些行列需要置零）。O(1) 空间的关键是**用矩阵的第一行和第一列作为标记数组**。

1. 先单独记录第一行、第一列本身是否含零
2. 遍历 `[1..m-1][1..n-1]`，如果 `matrix[i][j] == 0`，把 `matrix[i][0]` 和 `matrix[0][j]` 标记为 0
3. 根据第一行第一列的标记，置零内部区域
4. 最后处理第一行和第一列本身

## 解法

```cpp
class Solution {
public:
    void setZeroes(vector<vector<int>>& matrix) {
        int m = matrix.size(), n = matrix[0].size();
        bool firstRowZero = false, firstColZero = false;
        for (int j = 0; j < n; j++) if (matrix[0][j] == 0) firstRowZero = true;
        for (int i = 0; i < m; i++) if (matrix[i][0] == 0) firstColZero = true;
        for (int i = 1; i < m; i++)
            for (int j = 1; j < n; j++)
                if (matrix[i][j] == 0) { matrix[i][0] = 0; matrix[0][j] = 0; }
        for (int i = 1; i < m; i++)
            for (int j = 1; j < n; j++)
                if (matrix[i][0] == 0 || matrix[0][j] == 0) matrix[i][j] = 0;
        if (firstRowZero) for (int j = 0; j < n; j++) matrix[0][j] = 0;
        if (firstColZero) for (int i = 0; i < m; i++) matrix[i][0] = 0;
    }
};
```

## 复杂度

- **时间**：O(m × n)
- **空间**：O(1)

## 关键点

1. 必须**先记录**首行首列是否含零，再用它们做标记，否则信息会被覆盖
2. 处理顺序：记录 → 标记内部 → 置零内部 → 最后处理首行首列
