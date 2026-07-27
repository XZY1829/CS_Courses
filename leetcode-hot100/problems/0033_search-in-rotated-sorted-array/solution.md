# 33. 搜索旋转排序数组 - 题解

## 思路

旋转后数组分为两段有序部分。二分时判断 mid 落在哪一段，再决定搜索方向。

- `nums[lo] <= nums[mid]`：左半有序 → 判断 target 在不在左半
- 否则：右半有序 → 判断 target 在不在右半

## 解法

```cpp
int search(vector<int>& nums, int target) {
    int lo = 0, hi = nums.size() - 1;
    while (lo <= hi) {
        int mid = lo + (hi - lo) / 2;
        if (nums[mid] == target) return mid;
        if (nums[lo] <= nums[mid]) {
            if (nums[lo] <= target && target < nums[mid]) hi = mid - 1;
            else lo = mid + 1;
        } else {
            if (nums[mid] < target && target <= nums[hi]) lo = mid + 1;
            else hi = mid - 1;
        }
    }
    return -1;
}
```

## 复杂度

- **时间**：O(log n)　　**空间**：O(1)

## 关键点

1. `nums[lo] <= nums[mid]` 用 `<=` 处理 lo==mid 的情况
2. 判断 target 是否在有序半段内时，两端都要检查
