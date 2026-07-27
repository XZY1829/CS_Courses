# 42. 接雨水 - 题解

## 思路

每个位置 i 能接的水 = `min(leftMax, rightMax) - height[i]`，其中 leftMax/rightMax 是 i 左/右侧的最大高度。

**双指针法**可以在 O(1) 空间内完成：

- 维护 `left`、`right` 指针，以及 `leftMax`、`rightMax`
- 当 `height[left] < height[right]` 时，左侧是瓶颈：
  - 更新 `leftMax`，当前位置能接 `leftMax - height[left]` 的水
  - `left++`
- 否则右侧是瓶颈，对称操作

**关键洞察**：当 `height[left] < height[right]` 时，rightMax 一定 ≥ height[right] > height[left]，所以 `min(leftMax, rightMax) = leftMax`，不需要知道真正的 rightMax。

## 解法

```cpp
class Solution {
public:
    int trap(vector<int>& height) {
        int left = 0, right = (int)height.size() - 1;
        int leftMax = 0, rightMax = 0;
        int water = 0;
        while (left < right) {
            if (height[left] < height[right]) {
                leftMax = max(leftMax, height[left]);
                water += leftMax - height[left];
                left++;
            } else {
                rightMax = max(rightMax, height[right]);
                water += rightMax - height[right];
                right--;
            }
        }
        return water;
    }
};
```

## 复杂度

- **时间**：O(n)
- **空间**：O(1)

## 其他解法

| 方法 | 时间 | 空间 | 特点 |
|------|------|------|------|
| 前缀最大值数组 | O(n) | O(n) | 最直观，预计算 leftMax[] 和 rightMax[] |
| 单调栈 | O(n) | O(n) | 横向计算水量，按层累加 |
| 双指针 | O(n) | O(1) | 空间最优，面试首选 |

## 关键点

1. 理解"每个位置的水量取决于两侧最大值中的较小值"是解题的核心
2. 双指针法的巧妙之处：移动较短边时，另一侧一定存在不小于当前边的高度
3. 面试高频题，建议三种解法都掌握
