# 34. 在排序数组中查找元素的第一个和最后一个位置 - 题解

## 思路

两次二分：`lower_bound(target)` 找左边界，`lower_bound(target+1) - 1` 找右边界。

## 解法

```cpp
class Solution {
    int lowerBound(vector<int>& nums, int target) {
        int lo = 0, hi = nums.size();
        while (lo < hi) { int mid = lo+(hi-lo)/2; if (nums[mid]<target) lo=mid+1; else hi=mid; }
        return lo;
    }
public:
    vector<int> searchRange(vector<int>& nums, int target) {
        int left = lowerBound(nums, target);
        if (left == nums.size() || nums[left] != target) return {-1,-1};
        return {left, lowerBound(nums, target+1) - 1};
    }
};
```

## 复杂度

- **时间**：O(log n)　　**空间**：O(1)

## 关键点

1. 右边界 = `lower_bound(target+1) - 1`，比单独写 `upper_bound` 更简洁
