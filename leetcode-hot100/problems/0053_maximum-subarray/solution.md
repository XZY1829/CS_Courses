# 53. 最大子数组和 - 题解

## 思路

**Kadane 算法**：维护"以当前元素结尾的最大子数组和" `curSum`。

对每个元素，有两个选择：
1. 接在前面的子数组后面：`curSum + nums[i]`
2. 重新开始一个新子数组：`nums[i]`

取较大者：`curSum = max(nums[i], curSum + nums[i])`

等价理解：如果 `curSum < 0`，前面的子数组是负贡献，不如丢弃重新开始。

## 解法

```cpp
class Solution {
public:
    int maxSubArray(vector<int>& nums) {
        int curSum = 0, maxSum = INT_MIN;
        for (int n : nums) {
            curSum = max(n, curSum + n);
            maxSum = max(maxSum, curSum);
        }
        return maxSum;
    }
};
```

## 复杂度

- **时间**：O(n)
- **空间**：O(1)

## 关键点

1. 全负数数组也能正确处理（选最大的那个负数），因为 `maxSum` 初始化为 `INT_MIN`
2. 这是一维 DP 的空间优化版本：`dp[i] = max(nums[i], dp[i-1] + nums[i])`
3. 如果要返回子数组本身（而非和），需要额外记录起止下标
