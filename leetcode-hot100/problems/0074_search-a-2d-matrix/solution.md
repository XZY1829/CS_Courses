# 74. 搜索二维矩阵 - 题解

## 思路

矩阵展开后是有序一维数组，直接二分。下标映射：`matrix[mid/n][mid%n]`。

## 解法

```cpp
bool searchMatrix(vector<vector<int>>& matrix, int target) {
    int m = matrix.size(), n = matrix[0].size();
    int lo = 0, hi = m * n - 1;
    while (lo <= hi) {
        int mid = lo + (hi - lo) / 2;
        int val = matrix[mid / n][mid % n];
        if (val == target) return true;
        else if (val < target) lo = mid + 1;
        else hi = mid - 1;
    }
    return false;
}
```

## 复杂度

- **时间**：O(log(m×n))　　**空间**：O(1)

## 关键点

1. 与第 240 题的区别：本题矩阵展开完全有序，可用纯二分；240 题只保证行列有序
