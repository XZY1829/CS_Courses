# 240. 搜索二维矩阵 II - 题解

## 思路

利用矩阵"行列均有序"的特性，从**右上角**开始搜索：

- `matrix[i][j] == target`：找到
- `matrix[i][j] > target`：当前值太大，向左走（`j--`），排除当前列
- `matrix[i][j] < target`：当前值太小，向下走（`i++`），排除当前行

右上角是唯一满足"向左变小、向下变大"的起点（左下角也可以，方向对称）。

## 解法

```cpp
class Solution {
public:
    bool searchMatrix(vector<vector<int>>& matrix, int target) {
        int m = matrix.size(), n = matrix[0].size();
        int i = 0, j = n - 1;
        while (i < m && j >= 0) {
            if (matrix[i][j] == target) return true;
            else if (matrix[i][j] > target) j--;
            else i++;
        }
        return false;
    }
};
```

## 复杂度

- **时间**：O(m + n)，每步排除一行或一列
- **空间**：O(1)

## 关键点

1. 从**右上角**或**左下角**出发——这两个点的行列方向单调性相反，可以做二叉搜索
2. 不能从左上角或右下角出发——两个方向都在增大（或减小），无法判断走哪个方向
3. 与第 74 题（搜索二维矩阵）的区别：74 题的矩阵展开后是完全有序的，可以用纯二分
