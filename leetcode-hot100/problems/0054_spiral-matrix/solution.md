# 54. 螺旋矩阵 - 题解

## 思路

**按层模拟**：维护四个边界 `top, bottom, left, right`，每一轮按「右→下→左→上」的顺序遍历一圈，然后收缩边界。

注意在"左"和"上"方向遍历前要检查边界是否仍然合法（`top <= bottom` / `left <= right`），避免单行或单列的情况下重复遍历。

## 解法

```cpp
class Solution {
public:
    vector<int> spiralOrder(vector<vector<int>>& matrix) {
        vector<int> result;
        int top = 0, bottom = matrix.size() - 1;
        int left = 0, right = matrix[0].size() - 1;
        while (top <= bottom && left <= right) {
            for (int j = left; j <= right; j++) result.push_back(matrix[top][j]);
            top++;
            for (int i = top; i <= bottom; i++) result.push_back(matrix[i][right]);
            right--;
            if (top <= bottom)
                for (int j = right; j >= left; j--) result.push_back(matrix[bottom][j]);
            bottom--;
            if (left <= right)
                for (int i = bottom; i >= top; i--) result.push_back(matrix[i][left]);
            left++;
        }
        return result;
    }
};
```

## 复杂度

- **时间**：O(m × n)
- **空间**：O(1)（不计输出）

## 关键点

1. "左"和"上"方向前需要额外边界检查——否则单行 `[[1,2,3]]` 或单列矩阵会重复输出
2. 模拟题的要点是边界处理严谨，建议手动模拟一个 2×3 的例子验证
