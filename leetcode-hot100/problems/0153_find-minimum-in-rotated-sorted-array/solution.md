# 153. 寻找旋转排序数组中的最小值 - 题解

## 思路

二分：比较 `nums[mid]` 和 `nums[hi]`。
- `nums[mid] > nums[hi]`：最小值在右半 → `lo = mid + 1`
- 否则：最小值在左半（包含 mid）→ `hi = mid`

## 解法

```cpp
int findMin(vector<int>& nums) {
    int lo = 0, hi = nums.size() - 1;
    while (lo < hi) {
        int mid = lo + (hi - lo) / 2;
        if (nums[mid] > nums[hi]) lo = mid + 1;
        else hi = mid;
    }
    return nums[lo];
}
```

## 复杂度

- **时间**：O(log n)　　**空间**：O(1)

## 关键点

1. 和 `nums[hi]` 比较而非 `nums[lo]`——避免未旋转数组的边界问题
