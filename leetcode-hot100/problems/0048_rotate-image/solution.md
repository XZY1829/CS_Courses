# 48. 旋转图像 - 题解

## 思路

顺时针旋转 90° = **转置 + 水平翻转**。

`(i,j) → 转置 → (j,i) → 水平翻转 → (j, n-1-i)`

验证：顺时针 90° 的目标位置就是 `(i,j) → (j, n-1-i)`，正确。

## 解法

```cpp
class Solution {
public:
    void rotate(vector<vector<int>>& matrix) {
        int n = matrix.size();
        // 转置：沿主对角线交换
        for (int i = 0; i < n; i++)
            for (int j = i + 1; j < n; j++)
                swap(matrix[i][j], matrix[j][i]);
        // 水平翻转：每行 reverse
        for (int i = 0; i < n; i++)
            reverse(matrix[i].begin(), matrix[i].end());
    }
};
```

## 复杂度

- **时间**：O(n²)
- **空间**：O(1)

## 关键点

1. 转置时注意 `j = i+1` 开始（只交换上三角），否则交换两次等于没交换
2. 逆时针 90° = 转置 + 垂直翻转（上下 reverse）
3. 旋转 180° = 水平翻转 + 垂直翻转
