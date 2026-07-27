# 1. 两数之和 - 题解

## 思路

暴力法需要 O(n²)，但题目要求找的是 `nums[i] + nums[j] == target`，即对每个 `nums[i]`，需要快速判断 `target - nums[i]` 是否已出现过。

**哈希表一次遍历**：边遍历边建立「值 → 下标」的映射。对当前元素 `nums[i]`，在哈希表中查找 `target - nums[i]`：
- 找到了：直接返回两个下标
- 没找到：把当前元素存入哈希表，继续遍历

因为先查后存，所以不会出现"用自己匹配自己"的情况。

## 解法

```cpp
class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        unordered_map<int, int> seen; // 值 → 下标
        for (int i = 0; i < (int)nums.size(); i++) {
            int complement = target - nums[i];
            if (seen.count(complement)) {
                return {seen[complement], i};
            }
            seen[nums[i]] = i;
        }
        return {};
    }
};
```

## 复杂度

- **时间**：O(n)，一次遍历，哈希表查找 O(1)
- **空间**：O(n)，哈希表最多存 n 个元素

## 关键点

1. **先查后存**：保证不会用同一个元素匹配两次，且能处理 `[3,3], target=6` 这种重复元素的情况
2. 题目保证恰好有一个解，因此不需要处理无解的情况
3. 返回的是**下标**不是值，所以哈希表存的是 `值→下标` 的映射
