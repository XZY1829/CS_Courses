# 11. 盛最多水的容器 - 题解

## 思路

面积 = `min(height[l], height[r]) × (r - l)`。

**左右双指针**从两端向中间收缩：每次移动**较短**的那一边。

**正确性**：当 `height[l] < height[r]` 时，保持 l 不动而移动 r，宽度减小，高度受限于 `height[l]`，面积只会更小。因此以 l 为左端的所有组合都不可能更优，可以安全地 `l++`。

## 解法

```cpp
class Solution {
public:
    int maxArea(vector<int>& height) {
        int left = 0, right = (int)height.size() - 1;
        int ans = 0;
        while (left < right) {
            int area = min(height[left], height[right]) * (right - left);
            ans = max(ans, area);
            if (height[left] < height[right])
                left++;
            else
                right--;
        }
        return ans;
    }
};
```

## 复杂度

- **时间**：O(n)，双指针最多移动 n 次
- **空间**：O(1)

## 关键点

1. **移动较短边**的贪心策略是正确的——移动较长边不可能使面积增大
2. 当 `height[l] == height[r]` 时移动哪边都行，不影响最终结果
3. 注意和「接雨水」区分：此题求两条线围成的最大矩形面积，接雨水求所有凹槽的总水量
