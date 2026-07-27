# 41. 缺失的第一个正数 - 题解

## 思路

要求 O(n) 时间 + O(1) 空间，不能用额外的 HashSet。

**原地哈希**：把数组自身当哈希表——值 `v` 应该放在下标 `v-1` 的位置。

1. 遍历数组，对每个元素尝试将其交换到正确位置：`nums[i]` 应放到 `nums[nums[i]-1]`
2. 交换后继续检查当前位置（用 while 而非 if），直到当前位置的值已就位或不在有效范围 `[1,n]`
3. 最后扫一遍，第一个 `nums[i] != i+1` 的位置，`i+1` 就是答案

## 解法

```cpp
class Solution {
public:
    int firstMissingPositive(vector<int>& nums) {
        int n = nums.size();
        for (int i = 0; i < n; i++) {
            while (nums[i] > 0 && nums[i] <= n && nums[nums[i] - 1] != nums[i]) {
                swap(nums[i], nums[nums[i] - 1]);
            }
        }
        for (int i = 0; i < n; i++) {
            if (nums[i] != i + 1) return i + 1;
        }
        return n + 1;
    }
};
```

## 复杂度

- **时间**：O(n)，虽然有 while 循环，但每个元素最多被交换到正确位置一次
- **空间**：O(1)，原地操作

## 关键点

1. while 条件中 `nums[nums[i]-1] != nums[i]` 防止死循环（两个位置值相同时停止）
2. 答案一定在 `[1, n+1]` 范围内（n 个位置最多放 1~n）
3. 负数和超出范围的值不需要处理，它们会被留在原地，最终暴露出空缺位置
