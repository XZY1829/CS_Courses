# 35. 搜索插入位置 - 题解

## 思路

标准**左闭右开二分**，找第一个 ≥ target 的位置（即 `lower_bound`）。

## 解法

```cpp
int searchInsert(vector<int>& nums, int target) {
    int lo = 0, hi = nums.size();
    while (lo < hi) {
        int mid = lo + (hi - lo) / 2;
        if (nums[mid] < target) lo = mid + 1;
        else hi = mid;
    }
    return lo;
}
```

## 复杂度

- **时间**：O(log n)　　**空间**：O(1)

## 关键点

1. 本质就是 `lower_bound`
2. `hi` 初始化为 `nums.size()`（开区间），处理 target 大于所有元素的情况
